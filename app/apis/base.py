import asyncio
import gc
import pathlib
import subprocess

import rdkit
import torch
from fastapi import APIRouter, Body, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from funnel.configuration import THRESHOLD_SURROGATE
from funnel.generator.vocab import common_atom_vocab
from funnel.generator.wrappers import GeneratorModel
from funnel.surrogate.processor import QDFprocessor, smiles_to_xyz
from funnel.surrogate.wrappers import SurrogateModel
from rdkit import Chem
from rdkit.Chem import Draw


def count_electrons(smiles: str):
    """return the number of electrons in a SMILES string molecule"""
    cmd_string = f"obabel -:'{smiles}' -oxyz -h --gen3D > tmp.txt"
    subprocess.check_call(cmd_string, shell=True)
    with open("tmp.txt", "r") as f:
        nelec = f.readlines()[0].strip()
    return nelec


def funnel_half(
    pretrained_generator: str,
    pretrained_surrogate: str,
    motif_vocab: str,
    orbital_file: str,
    device_gen: str,
    device_surr: str,
    cond: float,
    num_samples: int,
):

    # load the wrappers of the generator and surrogate model
    device = torch.device(device_gen)
    generator = GeneratorModel(
        pretrained=pretrained_generator,
        motif_vocab=motif_vocab,
        atom_vocab=common_atom_vocab,
        device=device,
    )

    # generate the initial set
    generator.generate(num_samples=num_samples, condition=cond, unique=True)
    generated = generator.get_results()

    del generator
    collected = gc.collect()

    # dump the generated smiles into a file
    with open("gen_out.smi", "w") as f:
        [print(smile, file=f) for smile in generated]

    # transform the smiles file to xyz
    smiles_to_xyz(in_filename="gen_out.smi", out_filename="surr_in.xyz")

    # load the dataset with the xyz smiles
    dataset = QDFprocessor(orbital_file=orbital_file)
    dataset.process_from_file(filename="surr_in.xyz")

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        min(num_samples, 64),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: list(zip(*xs)),
        pin_memory=True,
    )

    del dataset
    collected = gc.collect()

    device = torch.device(device_surr)
    surrogate = SurrogateModel(
        pretrained=pretrained_surrogate,
        orbital_file=orbital_file,
        device=device,
    )

    # feed the data to the surrogate model
    for data in dataloader:
        _, _ = surrogate.forward(batch=data)

    # get the results from the internal buffers
    molecule_ids, predictions = surrogate.get_results()

    ctr_initial, ctr_final = 0, 0
    molecule_ids_curated, predictions_curated = [], []
    items_list = []
    for mol, pred in zip(molecule_ids, predictions):
        ctr_initial += 1
        if abs(pred - cond) < THRESHOLD_SURROGATE:
            ctr_final += 1
            molecule_ids_curated.append(mol)
            predictions_curated.append(pred)
            items_list.append({"smile": mol, "pred": round(pred, 3)})

    return {
        "molecule_ids": molecule_ids_curated,
        "predictions": predictions_curated,
        "condition": cond,
        "num_generated": ctr_final,
        "items": items_list,
    }


api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@api_router.get("/")
def home(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})


@api_router.get("/test")
def test(request: Request):
    with open("./static/sample.xyz", "r") as f:
        lines = f.read()
    return lines


@api_router.get("/test2")
def test2(request: Request):
    return templates.TemplateResponse("3dmol.html", {"request": request})


@api_router.get("/render/{smile}")
async def read_item(smile: str, gap: float, request: Request):

    """Look fot the structure associated to the SMILES string given."""

    # check if files created during Funnel pipeline exist
    if (
        pathlib.Path("./gen_out.smi").exists()
        and pathlib.Path("./surr_in.xyz").exists()
    ):

        # initialize flag
        smile_present = False

        # check SMILES is in the generated file
        with open("./gen_out.smi", "r") as f:
            lines = f.readlines()
            lines = [item.replace("\n", "") for item in lines]
            if smile in lines:
                smile_present = True

        # if SMILES present: parse the xyz file for the structure of the prompted SMILES
        if smile_present:
            with open("./surr_in.xyz", "r") as f:
                chunks = f.read().strip().split("\n\n")

            for chunk in chunks:
                # chunk header (first line) has the SMILES string
                chunk_head = chunk.split("\n")[0]

                # check for a match
                if smile == chunk_head:
                    with open("./static/mol2render.xyz", "w") as g:
                        nelec = count_electrons(smile)
                        print(nelec, file=g)
                        print("", file=g)
                        for line in chunk.split("\n")[1:]:
                            print(line.replace("\n", ""), file=g)

                    # create the image of the molecule
                    rdkit_molecule = Chem.MolFromSmiles(smile)
                    img = Draw.MolToImage(rdkit_molecule)
                    img.save("./static/drawing.png")

                    return templates.TemplateResponse(
                        "render.html",
                        {
                            "request": request,
                            "smile": smile,
                            "gap": gap,
                            "sample": chunk.split("\n")[1:],
                        },
                    )

        # else if SMILES not there, but generation files exist
        return templates.TemplateResponse(
            "notfound.html", {"request": request, "smile": smile, "gap": gap}
        )

    else:
        # else if no generation files were found (user didnt generate)
        return templates.TemplateResponse(
            "notfound.html", {"request": request, "smile": smile, "gap": gap}
        )


# dummpy route acting as a placeholder
@api_router.get("/no-content")
def notfound(request: Request):
    return templates.TemplateResponse("deadend.html", {"request": request})


@api_router.post("/generate")
def generate(request: Request, condition: float = Form(...)):
    """For rendering results on HTML GUI"""

    # verify user input is within the range of values of seen samples
    if (condition > 4.5) or (condition < 1.6):
        warning_message = f"Warning: HOMO-LUMO gap value given outside of range [1.6,4.5] eV. Try a value within the range."
        return templates.TemplateResponse(
            "homepage.html",
            context={"request": request, "warning_message": warning_message},
        )

    pretrained_generator = "./pretrained/pretrained_generator"
    pretrained_surrogate = "./pretrained/pretrained_surrogate"
    motif_vocab = "./configuration/vocab.txt"
    orbital_file = "./configuration/orbitaldict_6-31G.pickle"

    # runs the half pipeline (Generator + Surrogate)
    model_response = funnel_half(
        pretrained_generator=pretrained_generator,
        pretrained_surrogate=pretrained_surrogate,
        motif_vocab=motif_vocab,
        orbital_file=orbital_file,
        device_gen="cpu",
        device_surr="cpu",
        cond=condition,
        num_samples=30,
    )

    return templates.TemplateResponse(
        "prediction.html",
        context={
            "request": request,
            "condition": model_response["condition"],
            "molecule_ids": model_response["molecule_ids"],
            "predictions": model_response["predictions"],
            "num_generated": model_response["num_generated"],
            "items": model_response["items"],
        },
    )
