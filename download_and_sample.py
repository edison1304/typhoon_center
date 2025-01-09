import re
from datetime import datetime
import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from rasterio.transform import from_origin
import csv
from download_from_kma import get_response, wait_until_morning


def extract_region(lc_latlon_path, target_lat, target_lon, nc_file, pixel_range=200):
    """
    lc_latlon 파일을 사용하여 잘라낼 범위를 계산한 후, 주어진 nc_file에서 해당 영역을 추출합니다.
    범위가 이미지 경계를 벗어나는 경우 None을 반환합니다.

    Parameters:
    - lc_latlon_path (str): lc_latlon 파일 경로
    - target_lat (float): 중심 위도
    - target_lon (float): 중심 경도
    - nc_file (str): 데이터 추출할 nc 파일 경로
    - pixel_range (int): 중심으로부터 범위 (픽셀 수)

    Returns:
    - (region_lat, region_lon, data, True) if extraction is successful
    - None if the requested range exceeds image boundaries
    """
    # lc_latlon 파일 열기
    lc_dataset = xr.open_dataset(lc_latlon_path)
    lc_latitudes = lc_dataset['lat'].values
    lc_longitudes = lc_dataset['lon'].values

    # 중심 좌표에 가장 가까운 인덱스 찾기
    lat_diff = np.abs(lc_latitudes - target_lat)
    lon_diff = np.abs(lc_longitudes - target_lon)
    lat_idx, lon_idx = np.unravel_index(
        np.argmin(lat_diff + lon_diff), lc_latitudes.shape
    )

    # 요청한 범위
    requested_lat_start = lat_idx - pixel_range
    requested_lat_end = lat_idx + pixel_range
    requested_lon_start = lon_idx - pixel_range
    requested_lon_end = lon_idx + pixel_range

    # 잘라낼 범위 계산
    lat_start = max(lat_idx - pixel_range, 0)
    lat_end = min(lat_idx + pixel_range, lc_latitudes.shape[0])
    lon_start = max(lon_idx - pixel_range, 0)
    lon_end = min(lon_idx + pixel_range, lc_longitudes.shape[1])

    # 범위 초과 여부 확인
    if (lat_start != requested_lat_start or
        lat_end != requested_lat_end or
        lon_start != requested_lon_start or
        lon_end != requested_lon_end):
        # 범위가 이미지 경계를 벗어남
        lc_dataset.close()
        return None

    # lc_latlon 기준으로 위도 및 경도 추출
    region_lats = lc_latitudes[lat_start:lat_end, lon_start:lon_end]
    region_lons = lc_longitudes[lat_start:lat_end, lon_start:lon_end]

    # 대상 nc 파일 열기
    dataset = xr.open_dataset(nc_file)
    # 데이터 추출 (첫 번째 변수를 자동 선택)
    variable = list(dataset.data_vars)[0]  # 첫 번째 변수 이름 가져오기
    data = dataset[variable].isel(
        dim_y=slice(lat_start, lat_end),
        dim_x=slice(lon_start, lon_end)
    ).values

    dataset.close()
    lc_dataset.close()

    return region_lats, region_lons, data, True

def download_typhoon_data(typhoon_data, bands, extra_auth_keys, log_file_path, current_auth_key):
    extra_auth_keys_l = extra_auth_keys.copy()
    current_auth_key_l = current_auth_key
    """
    추출된 태풍 데이터를 사용하여 파일을 다운로드합니다.

    Parameters:
    - typhoon_data (list): 태풍별 데이터 리스트
    - bands (list): 다운로드할 밴드 리스트
    - extra_auth_keys (list): 여분의 인증 키 리스트
    - log_file_path (str): 로그 파일 경로
    """
    for typhoon in typhoon_data:
        for entry in typhoon['data']:
            date_str = entry['time'].strftime("%Y%m%d%H%M")

            # 인증 키가 모두 소진되고 마지막 키로도 403이 발생하면 루프를 중단하고 대기
            while True:
                extra_keys, current_key, IsFailed = get_response(date_str, bands, extra_auth_keys_l, log_file_path, current_auth_key_l)
                if IsFailed:
                    wait_until_morning(log_file_path)  # 오전 5시까지 대기
                    # 인증 키를 다시 초기화
                    extra_auth_keys_l = [
                                   "0YI3qOGZRpOCN6jhmeaTdA", "i0oHwLBGTj-KB8CwRp4_ow", "f3UaCxwZRou1GgscGRaLeA", "fvPt09e-R5Cz7dPXvheQTQ", 
                                   "KJ3VhCjASJid1YQowNiYdA", "0CdTvwYRTqmnU78GET6pnA","kH7frqXsQh6-366l7EIePQ", "1Fo6dIfFQK2aOnSHxXCtbg",
                                   "4kmdlf-4TSOJnZX_uO0jvw", "zNMybVrVT6yTMm1a1d-sXw", "nEKJPx8sSQyCiT8fLEkMug", "P6_4JNbzS_qv-CTW86v6-w", 
                                   "KjVfdQprSDC1X3UKa8gwyg", "mOEJ-yaTQ4qhCfsmk8OKCQ", "KEnMgMUCTH2JzIDFAqx9Qw", "f_x9c96ZTNC8fXPemSzQRA"]
                    current_auth_key_l = extra_auth_keys_l.pop(0)
                else:
                    extra_auth_keys_l = extra_keys
                    current_auth_key_l = current_key
                    break


def extract_typhoon_data(file_path):
    """
    태풍 데이터를 파일에서 추출하여 구조화된 리스트로 반환합니다.

    Parameters:
    - file_path (str): 태풍 데이터 파일 경로

    Returns:
    - typhoon_data (list): 태풍별 데이터 리스트
    """
    typhoon_data = []
    current_typhoon = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('66666'):
                if current_typhoon and current_typhoon['data']:
                    typhoon_data.append(current_typhoon)
                name = ''.join(filter(lambda x: not x.isdigit() and not x.isspace(), line))
                name = name if name else 'Unknown'
                current_typhoon = {
                    'name': name,
                    'data': []
                }
            else:
                # 데이터 라인 추출
                time_str = line[0:8]  # 'YYMMDDHH'
                grade = line[13]
                latitude = float(line[15:18]) / 10
                longitude = float(line[19:23]) / 10

                try:
                    # 'YYMMDDHH' 형식을 'YYYYMMDDHH00'으로 변환 ('YY'는 '20YY'로 간주)
                    analysis_time = datetime.strptime('20' + time_str, '%Y%m%d%H')
                    # 분 정보가 없으므로 '00'으로 설정
                    analysis_time = analysis_time.replace(minute=0)
                except ValueError:
                    print(f"Invalid time format: {time_str}")
                    continue

                if datetime(2019, 7, 1) <= analysis_time <= datetime(2024, 11, 18):
                    typhoon_entry = {
                        'time': analysis_time,
                        'grade': grade,
                        'latitude': latitude,
                        'longitude': longitude,
                    }
                    current_typhoon['data'].append(typhoon_entry)

        if current_typhoon['data']:
            typhoon_data.append(current_typhoon)

    return typhoon_data


def load_nc_files(nc_folder):
    """
    typhoon_data 폴더 내 모든 nc 파일을 로드하여 파일명을 키로 하는 딕셔너리 반환
    동일 시간의 파일들을 리스트로 저장
    'VI006'이 포함된 파일은 제외
    파일명에서 채널 정보도 추출하여 저장

    Parameters:
    - nc_folder (str): nc 파일들이 저장된 폴더 경로

    Returns:
    - nc_files (dict): 시간별 nc 파일 정보 딕셔너리
    """
    nc_files = {}
    # IR, SW 채널을 포함하도록 정규표현식 수정
    channel_pattern = re.compile(r'(IR|SW)\d{3}')  # 예: IR105, IR112, SW038 등

    for filename in os.listdir(nc_folder):
        if filename.endswith('.nc') and 'VI006' not in filename:
            # 파일명에서 시간과 채널 정보를 추출
            # 예시: '202005121500_SW038.nc' -> time_str='202005121500', channel='SW038'
            parts = filename.split('_')
            if len(parts) < 2:
                print(f"Invalid filename format: {filename}")
                continue
            time_str = parts[0]  # 'YYYYMMDDHHMM'
            channel_match = channel_pattern.search(parts[1])
            channel = channel_match.group() if channel_match else 'UNKNOWN'

            if time_str not in nc_files:
                nc_files[time_str] = []
            nc_files[time_str].append({
                'path': os.path.join(nc_folder, filename),
                'channel': channel
            })

    return nc_files


def save_to_tiff(merged_data, region_lat, region_lon, output_path, channels):
    """
    합쳐진 데이터를 TIFF 파일로 저장
    위도와 경도 정보를 추가로 포함
    채널 정보를 메타데이터로 저장

    Parameters:
    - merged_data (np.ndarray): 합쳐진 채널별 데이터
    - region_lat (np.ndarray): 위도 배열
    - region_lon (np.ndarray): 경도 배열
    - output_path (str): 저장할 TIFF 파일 경로
    - channels (list): 채널 이름 리스트
    """
    # 변환 설정
    lat_res = (region_lat[-1][0] - region_lat[0][0]) / len(region_lat)
    lon_res = (region_lon[0][-1] - region_lon[0][0]) / len(region_lon[0])
    transform = from_origin(region_lon[0][0], region_lat[0][0], lon_res, lat_res)

    # 위도와 경도를 각각 별도의 밴드로 추가
    lat_data = region_lat
    lon_data = region_lon

    # 위도와 경도를 데이터와 함께 스택
    merged_with_coords = np.vstack([
        merged_data,                # 채널별 데이터
        lat_data[np.newaxis, :, :], # 위도
        lon_data[np.newaxis, :, :]  # 경도
    ])

    total_bands = merged_with_coords.shape[0]

    # 채널 리스트에 위도, 경도 추가
    bands_description = channels + ['Latitude', 'Longitude']

    # TIFF 저장
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=merged_data.shape[1],
        width=merged_data.shape[2],
        count=total_bands,  # 밴드 수 (채널 수 + 위도 + 경도)
        dtype=merged_with_coords.dtype,
        transform=transform,
        crs='EPSG:4326'
    ) as dst:
        for i in range(total_bands):
            dst.write(merged_with_coords[i], i + 1)
            dst.set_band_description(i + 1, bands_description[i])


def main():
    file_path = 'bst_all.txt'
    nc_folder = 'typhoon_data'  # nc 파일들이 저장된 폴더 경로
    lc_latlon = 'gk2a_ami_ea020lc_latlon.nc'
    output_folder = 'sampled_data'  # TIFF 파일을 저장할 기본 폴더
    os.makedirs(output_folder, exist_ok=True)

    metadata = []  # 메타데이터를 저장할 리스트

    typhoon_data = extract_typhoon_data(file_path)
    nc_files = load_nc_files(nc_folder)

    for typhoon in typhoon_data:
        name = typhoon['name'] if typhoon['name'] else 'Unknown'
        for entry in typhoon['data']:
            # 'YYMMDDHH' -> '20YYMMDDHH00'
            time_str_full = entry['time'].strftime('%Y%m%d%H%M')  # 'YYYYMMDDHHMM'
            year = entry['time'].strftime('%Y')
            grade = entry['grade']
            print(time_str_full)
            matching_nc_files = nc_files.get(time_str_full)
            if matching_nc_files:
                region_lats = []
                region_lons = []
                data_list = []
                channels = []

                for nc_file in matching_nc_files:
                    extraction_result = extract_region(
                        lc_latlon_path=lc_latlon,       # lc_latlon 경로 전달
                        target_lat=entry['latitude'],    # 중심 위도
                        target_lon=entry['longitude'],   # 중심 경도
                        nc_file=nc_file['path'],         # 추출할 nc 파일 경로
                        pixel_range=200                   # 필요 시 조정
                    )
                    if extraction_result is not None:
                        region_lat, region_lon, data, success = extraction_result
                        if success:
                            region_lats.append(region_lat)
                            region_lons.append(region_lon)
                            data_list.append(data)
                            channels.append(nc_file['channel'])
                    else:
                        # 범위 초과로 인한 추출 실패
                        print("    범위 초과로 인해 데이터 추출을 건너뜁니다.")
                        break  # 현재 태풍의 해당 엔트리 처리를 중단

                # 모든 nc 파일에서 정상적으로 추출되었는지 확인
                if len(data_list) == len(matching_nc_files):
                    # 데이터 합치기 (채널을 첫 번째 차원으로 스택)
                    merged_data = np.stack(data_list)

                    # 저장 경로 설정: output_folder/year/name/time.tif
                    typhoon_folder = os.path.join(output_folder, year, name)
                    os.makedirs(typhoon_folder, exist_ok=True)
                    output_filename = f"{time_str_full}.tif"
                    output_path = os.path.join(typhoon_folder, output_filename)

                    # TIFF 저장
                    save_to_tiff(merged_data, region_lats[0], region_lons[0], output_path, channels)

                    # 메타데이터 추가
                    metadata.append({
                        'year': year,
                        'name': name,
                        'grade': grade,
                        'time': entry['time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'channels': ','.join(channels)
                    })

                    print(f"    saved: {output_path}")
                else:
                    print("    데이터 추출 실패로 인해 저장 및 메타데이터 생성을 건너뜁니다.")
            else:
                print("    no nc file")
        print()

    # 메타데이터 CSV 저장
    csv_path = os.path.join(output_folder, 'metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['year', 'name', 'grade', 'time', 'channels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in metadata:
            writer.writerow(data)

    print(f"metadata saved: {csv_path}")

def download():
    bands = ["IR105", "IR112", "IR123", "SW038"]
    extra_auth_keys = ["SQpr2nlTSm2Ka9p5U-ptLQ", "7Y0Uq8kqQOeNFKvJKvDn2g", "qRjwoc5gSGiY8KHOYMhoQw", "-n3ysiuWQ7698rIrltO-gg", 
                       "0YI3qOGZRpOCN6jhmeaTdA", "i0oHwLBGTj-KB8CwRp4_ow", "f3UaCxwZRou1GgscGRaLeA", "fvPt09e-R5Cz7dPXvheQTQ", 
                       "KJ3VhCjASJid1YQowNiYdA", "0CdTvwYRTqmnU78GET6pnA","kH7frqXsQh6-366l7EIePQ", "1Fo6dIfFQK2aOnSHxXCtbg",
                       "4kmdlf-4TSOJnZX_uO0jvw", "zNMybVrVT6yTMm1a1d-sXw", "nEKJPx8sSQyCiT8fLEkMug", "P6_4JNbzS_qv-CTW86v6-w", 
                       "KjVfdQprSDC1X3UKa8gwyg", "mOEJ-yaTQ4qhCfsmk8OKCQ", "KEnMgMUCTH2JzIDFAqx9Qw", "f_x9c96ZTNC8fXPemSzQRA"]
    log_file_path = "download_log.txt"
    current_auth_key = "0CdTvwYRTqmnU78GET6pnA"
    typhoon_data = extract_typhoon_data('bst_all.txt')
    download_typhoon_data(typhoon_data, bands, extra_auth_keys, log_file_path, current_auth_key)
    print("download completed")

if __name__ == "__main__":
    main()
