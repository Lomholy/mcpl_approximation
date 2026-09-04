import mcpl

myfile = mcpl.MCPLFile("ODIN.mcpl.gz")
# for p in myfile.particle_blocks:
    # print(p.x, p.y, p.z, p.ekin)

stats = mcpl.collect_stats(myfile)


mcpl.plot_stats(stats)
