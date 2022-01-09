from pytrials.client import ClinicalTrials
import geopy
from geopy.geocoders import Nominatim, DataBC, GeoNames, Photon, Bing
import pandas as pd
import numpy as np
from time import sleep
from func_timeout import func_timeout, FunctionTimedOut


def query(key):
    ct = ClinicalTrials()
    search = ct.get_study_fields(
        search_expr=key,
        fields=[
            "OfficialTitle",
            "OverallStatus",
            "StartDate",
            "BriefSummary",
            # "DetailedDescription",
            "NCTId",
            # "StudyType",
            # "PhaseList",
            "LocationFacility",
            ],
        max_studies=20
    )
    return search


def load():
    geolocator_bing = Bing(api_key="Amh-_uUK56C9ZaUCI63lVDVyJLRQGGOmgeNWVPrO0gu4YufXeCaNmOzOqmsZA-Vx")
    geolocator_nominatim = Nominatim(user_agent="Trial-AId")
    return geolocator_bing, geolocator_nominatim


def geocode(address, geolocator_bing, geolocator_nominatim):
    try:
        location = geolocator_bing.geocode(address)
        if location is None:
            location = geolocator_nominatim.geocode(address)
            if location is None:
                sleep(1)
                return np.nan, np.nan
            else:
                sleep(1)
                return location.latitude, location.longitude
        else:
            sleep(1)
            return location.latitude, location.longitude
    except geopy.adapters.AdapterHTTPError as e:
        print(e)
        location = geolocator_nominatim.geocode(address)
        if location is None:
            sleep(1)
            return np.nan, np.nan
        else:
            sleep(1)
            return location.latitude, location.longitude
    except geopy.exc.GeocoderQueryError as e:
        print(e)
        location = geolocator_nominatim.geocode(address)
        if location is None:
            sleep(1)
            return np.nan, np.nan
        else:
            sleep(1)
            return location.latitude, location.longitude


def get_dataframe(key):
    while True:
        try:
            search = func_timeout(5, query, args=(key,))
            break
        except FunctionTimedOut:
            print("TIMEOUT")
            continue
    df = pd.DataFrame.from_records(search[1:], columns=search[0])
    df = df[df.OverallStatus != "Completed"]
    df = df[df.LocationFacility != ""]
    df["Link"] = df["NCTId"].apply(lambda x: "https://clinicaltrials.gov/ct2/show/" + x)
    df.drop(columns=["Rank", "NCTId"], inplace=True)
    return df


def get_coordinates(df):
    geolocator_bing, geolocator_nominatim = load()
    df["Latitude"] = df["LocationFacility"]\
        .apply(lambda x: geocode(x, geolocator_bing, geolocator_nominatim)[0])
    df["Longitude"] = df["LocationFacility"]\
        .apply(lambda x: geocode(x, geolocator_bing, geolocator_nominatim)[1])
    df.dropna(inplace=True)
    # return df[["Longitude", "Latitude"]]
    return df



if __name__ == "__main__":
    df = get_dataframe("LSD")
    print(df)
    df = get_coordinates(df)
    print(df)



# import json
# with open("seach_results_specific.json", "w") as f:
#     f.write(json.dumps(search))
