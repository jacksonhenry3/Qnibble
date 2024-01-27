import h5py
import os

# get list of all files in directory
directory = "../data/testing_gdf5"


def merge_groups(group1, group2, group_out):
    # Merge datasets
    for ds in group1.keys():
        if isinstance(group1[ds], h5py.Dataset):
            if ds not in group_out:  # Avoid overwriting datasets present in both files
                group1.copy(ds, group_out)

    # Merge groups
    for gr in group1.keys():
        if isinstance(group1[gr], h5py.Group):
            if gr not in group2:
                group_out.copy(group1[gr], gr)
            else:
                # Check if group already exists in the output file
                if gr not in group_out:
                    group_out.create_group(gr)
                merge_groups(group1[gr], group2[gr], group_out[gr])


def merge_hdf5_files(directory):
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".hdf5"):
            file_list.append(os.path.join(directory, filename))
        else:
            continue

    # extract the last part of the directory name
    output_file = directory + "/" + directory.split('/')[-1] + '.hdf5'
    with h5py.File(output_file, 'w') as fo:
        for file in file_list:
            with h5py.File(file, 'r') as fi:
                merge_groups(fi, fo, fo)

    #     close the files
    fo.close()
