import mcstasscript as ms
import os
import copy

for file in os.listdir("../../data_files/simulations/"):
    try:
        os.rmdir("../../data_files/simulations/" + file)
    except Exception as e:
        print("Did not delete " + {file})


instr = ms.McStas_instr(
    "odin_sample", author="Daniel Lomholt Christensen", origin="UCPH @ NBI"
)

mcstas_file = ms.McStas_file("./odin_sample.instr")
mcstas_file.add_to_instr(instr)


instr.set_parameters(run_from_mcpl=f'"../../data_files/mcpl_files/ODIN_n_11_2.mcpl.gz"')
instr.settings(output_path="../../data_files/simulations/raw_2") 
instr.backengine()

instr.set_parameters(run_from_mcpl=f'"../../data_files/mcpl_files/ODIN_n11.mcpl.gz"')
instr.settings(output_path="../../data_files/simulations/raw_1") 
instr.backengine()

for file in os.listdir("mcpl_files"):
    if not file.endswith(".mcpl.gz"):
        continue
    instr.set_parameters(run_from_mcpl=f'"../../data_files/mcpl_files/{file}"')
    name = copy.copy(file)
    name = name.strip(".mcpl.gz")
    instr.settings(output_path="simulations/" + name) 
    instr.backengine()
    instr.settings(force_compile=False)
