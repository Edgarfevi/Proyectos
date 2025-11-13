from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder="/home/edgar/GitHub/Proyectos/IA/Imagenes_temp/almacen_img", dpi=300):
    """
    Convert a PDF file to images, one image per page.

    :param pdf_path: Path to the input PDF file.
    :param output_folder: Folder where the output images will be saved.
    :param dpi: Resolution of the output images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save each image
    for i, image in enumerate(images, start=1):
        image_path = os.path.join(output_folder, f'page_{i:02d}.png')
        image.save(image_path, 'PNG')
        print(f'Saved: {image_path}')


