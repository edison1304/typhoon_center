import time
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.exceptions import ChunkedEncodingError

#서버에 응답 요청, 시간, 밴드
def get_response(date, bands, extra_auth_keys, log_file, current_auth_key):

    for band in bands: 
        filename = f"typhoon_data/{date}_{band}.nc"
        
        # 파일이 이미 존재하는 경우, 다운로드를 건너뜀
        if os.path.exists(filename):
            log_message = f"{filename} already downloaded, skip\n"
            print(log_message, end = '')       
            status_code = 200
            continue

        
        url = f'https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{band}/EA/data?date={date}&authKey={current_auth_key}'
        status_code = download_file(url, filename, log_file)
        
        # 403 에러(api 제한) 발생 시, 여분의 인증 키가 있으면 순차적으로 시도
        while status_code == 403:
            if len(extra_auth_keys) > 0:  # 여분의 키가 있을 때만 교체
                log_message = f"403 error detected for {band} on {date}. Replacing try.\n"
                print(log_message, end = '')
                with open(log_file, 'a') as log:
                    log.write(log_message)

                current_auth_key = extra_auth_keys.pop(0)  # 여분의 키 중 하나를 사용                
                # 새 인증 키로 다시 시도
                url = f'https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{band}/EA/data?date={date}&authKey={current_auth_key}'
                status_code = download_file(url, filename, log_file)
                
                if status_code == 200:
                    log_message = f"Replacing Success {band} using new auth_key.\n"
                    print(log_message, end = '')
                    with open(log_file, 'a') as log:
                        log.write(log_message)
                    break

            else:
                # 모든 키 전부 소진 
                log_message = f"Successive Today, It is {datetime.now()} but No more key. Wait until morning.\n"
                print(log_message, end = '')
                with open(log_file, 'a') as log:
                    log.write(log_message)
                return True  # True를 반환하여 중단을 알림
                
    if status_code == 200:
        log_message = f"{date} download process completed.\n"
        print(log_message, end = '')
        with open(log_file, 'a') as log:
            log.write(log_message)
    
    return extra_auth_keys, current_auth_key, False  # 정상적으로 완료된 경우 False 반환

# 아침 5시까지 대기하는 함수 
def wait_until_morning(log_file):
    now = datetime.now()
    next_day = (now + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
    wait_seconds = (next_day - now).total_seconds()

    log_message = f"All authentication keys exhausted. Waiting until {next_day.strftime('%Y-%m-%d %H:%M:%S')} (approximately {wait_seconds / 3600:.2f} hours).\n"
    print(log_message, end = '')
    with open(log_file, 'a') as log:
        log.write(log_message)
    
    time.sleep(wait_seconds)

#파일 다운로드 
def download_file(url, filename, log_file, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024*1024*4):
                    if chunk:  # 데이터가 있을 경우에만 쓰기
                        file.write(chunk)

            if retries and response.status_code == 200:
                log_message = f"After retrying... Successdownload {url}.\n"
                print(log_message, end = '')
                with open(log_file, 'a') as log:
                    log.write(log_message)


            return response.status_code
        
        except ChunkedEncodingError as e: # 다운로드 오류 에러
            retries += 1
            log_message = f"ChunkedEncodingError occurred. Retrying... ({retries}/{max_retries})\n"
            print(log_message, end = '')
            with open(log_file, 'a') as log:
                log.write(log_message)
            
            time.sleep(200*retries+20)

        except requests.exceptions.HTTPError as e:
            # HTTP 에러 발생 시 상태 코드를 반환하고 재시도하지 않음

            if response.status_code == 403 or response.status_code == 404:
                log_message = f"HTTPError occurred: {e}, Status code: {response.status_code}. Returning status code.\n"
                print(log_message, end = '')
                with open(log_file, 'a') as log:
                    log.write(log_message)

                return response.status_code


            if retries < max_retries: # 403, 404 이외의 에러 발생
                retries += 1
                log_message = f"TMR occured, retrying....\n"
                print(log_message, end = '')
                with open(log_file, 'a') as log:
                    log.write(log_message)
                time.sleep(60*retries + 120)
                
            else:
                return response.status_code

        except requests.exceptions.RequestException as e:  # 인터넷 에러 
            retries += 1
            log_message = f"Error occurred: {e}. Retrying... ({retries}/{max_retries})\n"
            print(log_message, end = '')
            with open(log_file, 'a') as log:
                log.write(log_message)
            
            time.sleep(60 * retries + 20)




    log_message = f"Failed to download {filename} after {max_retries} retries.\n"
    print(log_message, end = '')
    with open(log_file, 'a') as log:
        log.write(log_message)

    return None

