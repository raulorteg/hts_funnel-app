import multiprocessing
import subprocess
import traceback
from pathlib import Path

import numpy as np


class QueueManager(object):
    def __init__(self, smiles_inputs: list, N_CPUs: int = 1, num_samples: int = 10):

        self.num_samples = min(num_samples, len(smiles_inputs))
        # idxs = np.random.randint(0, len(smiles_inputs) - 1, size=self.num_samples)
        # self.selected_smiles = [smiles_inputs[idx][0] for idx in idxs]
        self.selected_smiles = smiles_inputs
        self.N_CPUs = min(max(multiprocessing.cpu_count() - 1, 1), N_CPUs)

        # create the output directory
        Path("./outputs").mkdir(parents=True, exist_ok=True)
        self.output_dir = Path("./outputs")

        # some warning printing
        if (N_CPUs == 1) and (multiprocessing.cpu_count() > 1):
            print(
                "\t Note: this script can run in multiple CPUs,\
                consider running on more than 1 to speed up computations."
            )

    def _run_single(self, smiles: str):  # -> tuple(str, float):

        try:
            # create the xyz coordinates using openbabel
            outfile = Path(self.output_dir, f"{str(hash(smiles))}.inp")
            print(outfile)
            cmd_string = f"obabel -:'{smiles}' -oxyz -h --gen3D > {str(outfile)}"
            print(cmd_string)

            # try:
            # subprocess.call(cmd_string, shell=True, timeout=5*60)
            subprocess.check_call(cmd_string, shell=True)

            # to catch timeouts
            # except subprocess.TimeoutExpired:
            #    print(f"\t Timeout: Molecule {smiles} timeout out after {5*60} s.")

            # format the xyz file to be fed as input for orca
            # read the lines
            lines = []
            # lines.append("! PBE0 def2-TZVP RIJCOSX def2/j")
            # lines.append("! B3LYP def2-TZVP RIJCOSX def2/j PAL8") # suggestion 1 by bardi
            # lines.append("! B3LYP def2-TZVP OPT RIJCOSX SARC/J PAL8") # suggestion 2 by bardi
            lines.append("! BP86 def2-SVP PAL8")  # from the paper
            lines.append("*xyz 0 1")
            with open(outfile, "r") as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        lines.append(line.strip())
            lines.append("*")

            # rewrite them using the orca-expected format
            with open(outfile, "w") as f:
                for line in lines:
                    print(line, file=f)

            # use orca to perform the computations on the molecule
            infile = Path(self.output_dir, f"{str(hash(smiles))}.inp")
            outfile = Path(self.output_dir, f"{str(hash(smiles))}.out")
            cmd_string = f"~/orca/./orca {infile} > {outfile}"
            # cmd_string = f"mpirun -n 4 ~/orca/./orca {infile} > {outfile}" # mpirun -n 2 mpi-hello-world
            subprocess.check_call(cmd_string, shell=True)

            # parse the orca output to extract the homo-lumo gap
            with open(outfile, "r") as f:
                start_detected = False
                lines_since_start = 0
                homo_energy_ev, lumo_energy_ev = None, None

                for line in f:
                    # print(line)
                    # raise flag to start listening
                    if "ORBITAL ENERGIES" in line:
                        start_detected = True

                    # create a delay in the listening, so we skip the headers
                    if start_detected and (lines_since_start <= 3):
                        lines_since_start += 1

                    # if listening and not headers then start looking for the homo-lumo gap
                    elif start_detected and (lines_since_start > 3):
                        line = line.strip()
                        _, occupied, _, energy_ev = line.split()

                        # update the homo value to latest occupied occupied orbital energy
                        if float(occupied) in [1.0, 2.0]:
                            homo_energy_ev = float(energy_ev)

                        # set the lumo gap to the energy of the first unoccupoed orbital energy
                        if float(occupied) == 0.0:
                            lumo_energy_ev = float(energy_ev)
                            break

            if (not homo_energy_ev) or (not lumo_energy_ev):
                raise Exception(
                    f"Either homo or lumo or both values could not get \
                    extracted from the parsed script."
                )

            homo_lumo_gap_ev = lumo_energy_ev - homo_energy_ev
            return (smiles, homo_lumo_gap_ev)

        except Exception:
            print(f"\t Error: Molecule {smiles} produced an error.")
            traceback.print_exc()

    def run(self):  # -> list(tuple(str, float)):
        print(f"\t Initializing {self.N_CPUs} threads ...")
        pool = multiprocessing.Pool(processes=self.N_CPUs, maxtasksperchild=1)
        results = pool.map(
            func=self._run_single, iterable=self.selected_smiles, chunksize=1
        )
        pool.close()
        pool.join()

        return results


if __name__ == "__main__":

    smiles_inputs = [
        "c1c[nH]c(-c2cnc(-c3sc(-c4ccc[nH]4)c4nccnc34)c3nsnc23)c1",
        "c1c[nH]c(-c2ncc(-c3ccc(-c4scc5sccc45)s3)c3nsnc23)c1",
        "C1=C(c2cccc3c2=C[SiH2]C=3)Cc2c1c1cocc1c1c2[se]c2ccoc21",
        "C1=CCC(c2cnc(-c3ccc(-c4ccc[se]4)[se]3)c3nsnc23)=C1",
        "C1=CCC(c2cc3c(c4c2=C[SiH2]C=4)-c2[nH]c4cc[nH]c4c2C3)=C1",
        "C1=c2c(-c3ccco3)cc3c4nsnc4c4c5[nH]ccc5[nH]c4c3c2=C[SiH2]1",
        "C1=Cc2c(csc2-c2ccc(-c3ccc(-c4ccco4)o3)c3nsnc23)C1",
        "C1=Cc2cnc3c4c(c5nsnc5c3c2C1)C=C(c1ccc[nH]1)[SiH2]4",
        "C1=Cc2c(csc2-c2ccc(-c3cnc(-c4scc5[se]ccc45)s3)s2)[SiH2]1",
        "C1=C(c2scc3sccc23)[SiH2]c2c1ccc1c2ncc2cccnc21",
        "C1=c2c(-c3ccc[se]3)cc3cc4[nH]c5cc[nH]c5c4cc3c2=CC1",
        "c1cnc2cc(-c3cc4occc4c4c[nH]cc34)ccc2c1",
        "C1=C[SiH2]C(c2ccc3cccnc3c2)=C1",
        "C1=CCC(c2cnc3cccnc3c2)=C1",
        "c1ccc(-c2cnc3c(ccc4c5cnccc5oc43)c2)nc1",
        "c1cncc(-c2cc3[nH]c4c(ccc5ccoc54)c3cn2)c1",
        "c1ccc(-c2ccc3cccnc3c2)nc1",
        "c1cncc(-c2cc3occc3cn2)c1",
        "c1ccc(-c2ccc3ccoc3c2)cc1",
    ]
    queuer = QueueManager(smiles_inputs=smiles_inputs, N_CPUs=1, num_samples=1)
    results = queuer.run()
    f = open("out_dft.txt", "a+")
    for result in results:
        print(result, file=f)
    print(results)
    f.close()
