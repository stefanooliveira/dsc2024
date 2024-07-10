# cropper.py
import requests

RANGE = 5.0 #km

def download_sat_image(url: str, icao_code: str):
    response = requests.get(url)
    #plot image with opencv
    #try except and logging errors and exceptions
    read_image_from_response_and_cropit(response, icao_code)


def read_image_from_response_and_cropit(response: requests.Response, icao_code: str):
    latitude, longitude = get_airport_location_by_icao(icao_code)
    mask = get_mask_points_from_location(latitude, longitude)  
    # Open image from response
    # Crop image using mask
    # Return image cropped by location and range  

def get_mask_points_from_location(latitude: float, longitude: float, range_km: float):
    ''' 
    Receives a latitude and longitude and returns a List[Tuple[int, int]] as
    mask (in pixels) to be cropped. The mask is a square with 'range' with and height.
    '''
    
    pass

def get_airport_location_by_icao(icao_code: str) -> tuple[float, float]:
    '''
    Receives a ICAO code and returns its latitude and longitude
    '''
    pass



if __name__ == '__main__':
    download_sat_image("http://satelite.cptec.inpe.br/repositoriogoes/goes16/goes16_web/ams_ret_ch11_baixa/2022/06/S11635384_202206010100.jpg")

