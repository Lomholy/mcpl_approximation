import mcstasscript as ms
import os
import copy

instr = ms.McStas_instr(
    "odin_sample", author="Daniel Lomholt Christensen", origin="UCPH @ NBI"
)


mcstas_file = ms.McStas_file("./odin_sample.instr")

mcstas_file.add_to_instr(instr)


instr.set_parameters(run_from_mcpl=f'"../ODIN_n_11_2.mcpl.gz"')
instr.settings(output_path="simulations/raw_2") 

instr.backengine()
instr.set_parameters(run_from_mcpl=f'"../ODIN_n11.mcpl.gz"')
instr.settings(output_path="simulations/raw_1") 
instr.backengine()

for file in os.listdir("mcpl_files"):
    if not file.endswith(".mcpl.gz"):
        continue
    instr.set_parameters(run_from_mcpl=f'"mcpl_files/{file}"')
    name = copy.copy(file)
    name = name.strip(".mcpl.gz")
    instr.settings(output_path="simulations/" + name) 
    instr.backengine()
    instr.settings(force_compile=False)
