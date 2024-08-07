import os
import pandas as pd
import nibabel as nib

def get_nii_shapes(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            filepath = os.path.join(directory, filename)
            img = nib.load(filepath)
            header = img.header
            data.append({
                'filename': os.path.splitext(filename)[0],
                'x_dim': header.get_data_shape()[0],
                'y_dim': header.get_data_shape()[1],
                'z_dim': header.get_data_shape()[2]
            })
    return data

def save_to_dataframe(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    directory = "/home/ec2-user/SageMaker/shared/workstream-2/images_3d"
    output_file = "sizes.csv"
    nii_shapes = get_nii_shapes(directory)
    save_to_dataframe(nii_shapes, output_file)

