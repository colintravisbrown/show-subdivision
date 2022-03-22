# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc, StringArg
from chimerax.core.commands import *
from chimerax.std_commands import *
from chimerax.core.commands import all_objects
from chimerax.core.commands import run
from chimerax.atomic.molsurf import MolecularSurface
from chimerax.atomic import Atoms
import numpy as np
import matplotlib as mpl


# from chimerax.core.commands import measure


def show_subdivision(session, pdb, n):
    from numpy.linalg import norm
    # All command functions are invoked with ``session`` as its
    # first argument.  Useful session attributes include:
    #   logger: chimerax.core.logger.Logger instance
    #   models: chimerax.core.models.Models instance

    basepath = 'C:/Users/colin/OneDrive - San Diego State University (SDSU.EDU)/Research/Domain_Subdivision/mechanical_subdivision_ProDy/results/subdivisions/' + pdb + '/' + pdb + '_' + n + '_domains'
    path = basepath + '.pdb'

    run(session, 'open ' + '\'' + path + '\'')



    # center of mass measurement from ChimeraX tutorial
    atoms = all_objects(session).atoms  # getting atom list
    print(atoms)
    coords = atoms.scene_coords  # getting atom coords
    modelCenter = coords.mean(axis=0)  # calculate center of mass
    print("modelCenter", modelCenter)
    if modelCenter.any:
        print('Aligning Models')
        run(session, 'hkcage 1 0')
        modelX, modelY, modelZ = modelCenter
        run(session, 'move x ' + str(-1 * modelX) + ' coordinateSystem #1 models #1')  # adjust x coordinates
        run(session, 'move y ' + str(-1 * modelY) + ' coordinateSystem #1 models #1')  # adjust y coordinates
        run(session, 'move z ' + str(-1 * modelZ) + ' coordinateSystem #1 models #1')  # adjust z coordinates
        run(session, 'close #3')
    else:
        print('Models Already Aligned')

    atoms = all_objects(session).atoms  # acquire the atom list
    modelCenter = coords.mean(axis=0)  # calculate center of mass
    print("modelCenter", modelCenter)
    coords = atoms.scene_coords  # acquire atom coords
    labels = []
    for atom in atoms:
        labels.append(atom.bfactor)
    radius = norm(coords, axis=1).max()  # calculate the norms of the x coordinates, and then choose the maximum value
    print("radius_est: ", radius)

 #   coord_std_dev = np.std(coords)  # calculate the overall standard deviation of all coordinates
 #   print("coord std_dev: ", coord_std_dev)
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    cmap = generate_colormap(int(np.max(labels)))
    rgba = cmap(norm(labels))*255

# place a lattice on the structure. currently uses calculated radius, but other parameters are hard coded


    for i in range(len(atoms)):
        atom = atoms[i]
        atom.color = rgba[i,:]
        atom.name = str(i)

    # from chimerax.surface.surfacecmds import surface
    # for i in range(int(n)):
    #     ind =np.nonzero(np.array(labels,dtype=int)==i)[0]
    #     for j in range(len(ind)):
    #         atoms[ind[j]].selected = True
    #     run(session, 'surface sel')
    #
    #     for j in range(len(ind)):
    #         atoms[ind[j]].selected = False
    #
    #     r1 = rgba[i][0]
    #     g1 = rgba[i][1]
    #     b1 = rgba[i][2]
    #     run(session, 'color'  + '#1.' + str(r1) + ',' + str(g1) + ',' + str(b1) + ' target s')


    run(session, 'hkcage 1 0 alpha hexagonal-dual radius ' + str(radius) + ' spherefactor 0.2')
    run(session, 'view orient')
    run(session, 'style #1 sphere')
    run(session, 'save ' + '\'' + basepath + '.cxs' '\'')

# CmdDesc contains the command description.  For the
# "hello" command, we expect no arguments.

capsid_image_desc = CmdDesc( required=[('pdb', StringArg), ('n', StringArg)])





def generate_colormap(number_of_distinct_colors: int = 80):
    import math

    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import hsv

    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)
