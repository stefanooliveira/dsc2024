# cropper.py
import sys
import requests
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List

from . import get_icao_locations


PILCropMask = Tuple[int, int, int, int]

RANGE = 500  # km
# Map limits (in pixels) for sat image cropping (useful area only)
MAP_LIM_X0, MAP_LIM_Y0, MAP_LIM_X1, MAP_LIM_Y1 = 45, 107, 2174, 2235
# Map limits (in coordinates)
MAP_LAT_MIN, MAP_LAT_MAX = -56, 35
MAP_LON_MIN, MAP_LON_MAX = -116, -25


def download_sat_image(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        image_bytes = response.content
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be decoded.")
        return image
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error decoding image: {e}")
        sys.exit(1)


def crop_airport_area(full_image: cv2.typing.MatLike, icao_code: str, range_km):
    map_image = crop_map_limits(full_image, MAP_LIM_X0, MAP_LIM_Y0, MAP_LIM_X1, MAP_LIM_Y1)

    latitude, longitude = get_airport_location_by_icao(icao_code)
    
    mask = get_mask_from_location(map_image, latitude, longitude, range_km)
    roi = crop_image_with_mask(map_image, mask)
    
    # Convert numpy array to PIL Image
    roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Ensure correct color conversion
    
    # HACK            
    test_image_type(roi, "cropper crop_airport_area")
    
    return roi

def get_mask_from_location(map_image, latitude: float, longitude: float, range_km):
    ''' 
    Receives a latitude and longitude and returns a List[Tuple[int, int]] as
    mask (in pixels) to be cropped. The mask is a square with 'range' with and height.
    '''
    height, width, _ = map_image.shape
    airport_x, airport_y = coordinates_to_pixel_point(latitude, longitude,
                                                      width, height,
                                                      MAP_LAT_MIN, MAP_LAT_MAX,
                                                      MAP_LON_MIN, MAP_LON_MAX)
    
    square_coords = get_square_coordinates(airport_x, airport_y, range_km, width, height, latitude)

    mask = get_mask_from_square(square_coords, width, height)
    
    return mask

def get_square_coordinates(x, y, range_km, img_width, img_height, latitude):
    ''' Calculate the coordinates of the square given a center point (x, y) and range in km '''
    km_per_degree_lat = 111  # 1 degree of latitude is approximately 111 km
    km_per_degree_lon = 111 * np.cos(np.radians(latitude))  # adjust for longitude at given latitude
    
    # convert km to degrees
    range_deg_lat = range_km / km_per_degree_lat
    range_deg_lon = range_km / km_per_degree_lon

    # corvert range degree to pixels
    range_pixels_x = range_deg_lon / (MAP_LON_MAX - MAP_LON_MIN) * img_width / 2
    range_pixels_y = range_deg_lat / (MAP_LAT_MAX - MAP_LAT_MIN) * img_height / 2
    
    top_left = (x - range_pixels_x, y - range_pixels_y)
    top_right = (x + range_pixels_x, y - range_pixels_y)
    bottom_right = (x + range_pixels_x, y + range_pixels_y)
    bottom_left = (x - range_pixels_x, y + range_pixels_y)
    
    return [top_left, top_right, bottom_right, bottom_left]

def get_mask_from_square(square_coords, img_width, img_height):
    ''' Create a mask for cropping an image based on the square coordinates '''
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pts = np.array(square_coords, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask

def crop_image_with_mask(image, mask):
    ''' Crop the image using the provided mask '''
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    cropped_image = masked_image[y:y+h, x:x+w]
    return cropped_image

def coordinates_to_pixel_point(lat, lon, width, height, lat_min, lat_max, lon_min, lon_max):
    x = int((lon - lon_min) / (lon_max - lon_min) * width)
    y = int((lat_max - lat) / (lat_max - lat_min) * height)
    return x, y

def get_airport_location_by_icao(icao_code: str) -> Tuple[float, float]:
    ''' Receives a ICAO code and returns its latitude and longitude '''
    icao_locations = get_icao_locations()
    location = icao_locations.get(icao_code.upper())
    if location:
        return tuple(location)
    else:
        raise ValueError(f"Location for ICAO code {icao_code} not found.")

def crop_map_limits(image, lim_x0, lim_y0, lim_x1, lim_y1, save_path=None):
    # Convert image bytes to numpy
    cropped_image = image[lim_y0:lim_y1, lim_x0:lim_x1]

    if save_path:
        cv2.imwrite(save_path, cropped_image)

    return cropped_image

def test_image_type(image, location):
    if isinstance(image, np.ndarray):
        print(f"The image is a numpy.ndarray at {location}")
    elif isinstance(image, Image.Image):
        print(f"The image is a PIL.Image at {location}")
    else:
        print(f"Unknown image type at {location}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python cropper.py <codigo_aeroporto> <range>")
        sys.exit(1)
    
    codigo_aeroporto = sys.argv[1]
    range_km2 = sys.argv[2]

    full_sat_image = download_sat_image("http://satelite.cptec.inpe.br/repositoriogoes/goes16/goes16_web/ams_ret_ch11_baixa/2022/06/S11635384_202206010100.jpg")
    crop_airport_area(full_sat_image, codigo_aeroporto, int(range_km2))

# TODO Update cropper with v2 [OK]
# TODO Update to open icao_locations.json only once [OK]
# TODO Change images.py [OK]
# TODO Compare feature importance
# TODO Read paper
# TODO Make comparison