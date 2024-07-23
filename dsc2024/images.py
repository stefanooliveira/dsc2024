from typing import Tuple, List
from io import BytesIO

from dsc2024.datasets import get_image_mask_points
from dsc2024.roi_cropper import cropper

import requests

from PIL import Image as PIL_Image

PILCropMask = Tuple[int, int, int, int]


def generate_pil_crop_mask(mask_points: List[Tuple[int, int]]) -> PILCropMask:
    """Create from our four-point rectangle mask the PIL crop mask"""
    x = [x for x, _ in mask_points]
    y = [y for _, y in mask_points]
    return (min(x), min(y), max(x), max(y))

#TODO remake this entire function in cropper file
# def read_image_from_response_and_cropit(
#     response: requests.Response,
#     mask: PILCropMask
# ) -> PIL_Image.Image:
#     """Read from a successful response and generate a image cropped"""
#     img = PIL_Image.open(BytesIO(response.content))
#     img_cropped = img.crop(mask)
#     return img_cropped


def download_image_and_cropit(url: str, icao_code: str, range: int) -> PIL_Image.Image:
    full_sat_image = cropper.download_sat_image(url)
    return cropper.crop_airport_area(full_sat_image, icao_code, range)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import requests
    from PIL import Image
    from io import BytesIO
    
    test_icao = "SBSJ"
    test_range = 500

    cropper_image = download_image_and_cropit("http://satelite.cptec.inpe.br/repositoriogoes/goes16/goes16_web/ams_ret_ch11_baixa/2022/06/S11635384_202206010100.jpg",
                              test_icao, test_range)
    

            
    cropper.test_image_type(cropper_image, "main of images.py")

    # Download the image using requests
    response = requests.get(
        "http://satelite.cptec.inpe.br/repositoriogoes/goes16/goes16_web/ams_ret_ch11_baixa/2022/06/S11635384_202206010100.jpg"
    )
    img = Image.open(BytesIO(response.content))

    plt.figure(figsize=(15, 10))

    # First plot
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Second plot
    plt.subplot(1, 2, 1)
    plt.imshow(cropper_image)
    latitude = round(cropper.get_airport_location_by_icao(test_icao)[0],4)
    longitude = round(cropper.get_airport_location_by_icao(test_icao)[1],4)
    plt.title(f'Coordenadas para {test_icao}: ({latitude}, {longitude}) com {test_range}km^2')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.tight_layout()
    plt.show()