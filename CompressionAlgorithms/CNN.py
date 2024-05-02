import os
from pathlib import Path

project = os.path.join(os.path.expanduser("~"), "Project")


def process_file(filename):
    generate_map_command = f"cd {project}/CompressionAlgorithms/image-compression-cnn && python3 generate_map.py ../../Images/{filename}.tiff"
    os.system(generate_map_command)
    combine_images_command = f"cd {project}/CompressionAlgorithms/image-compression-cnn &&  python3 combine_images.py -image ../../Images/{filename}.tiff -map ./output/{filename}_map.jpg"
    os.system(combine_images_command)


def main():

    project = Path(str(os.getcwd()))
    folder_path =os.path.join(project, "Images")


    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file in files:
        # Process only .tiff files
        if file.endswith(".tiff") or file.endswith(".jpg"):
            filename = os.path.splitext(file)[0]
            process_file(filename)


if __name__ == "__main__":
    main()
