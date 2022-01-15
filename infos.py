from pytrials.client import ClinicalTrials
import geopy
from geopy.geocoders import Nominatim, DataBC, GeoNames, Photon, Bing
from geopy.adapters import AioHTTPAdapter
import pandas as pd
import numpy as np
from time import sleep, time
import asyncio


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
        max_studies=30,
        timer=1,
        retries=5
    )
    return search


async def geocode_async(address, timeout=10):

    try:
        async with Bing(
            api_key="Amh-_uUK56C9ZaUCI63lVDVyJLRQGGOmgeNWVPrO0gu4YufXeCaNmOzOqmsZA-Vx",
            adapter_factory=AioHTTPAdapter
            ) as geolocator_bing:
                location = await geolocator_bing.geocode(address, timeout=timeout)
        if location is None:
            async with Nominatim(
                user_agent="Trial-AId",
                adapter_factory=AioHTTPAdapter
                ) as geolocator_nominatim:
                location = await geolocator_nominatim.geocode(address, timeout=timeout)
        if location is None:
            return np.nan, np.nan
        else:
            return location.latitude, location.longitude

    except geopy.adapters.AdapterHTTPError as e:
        print(e)
        async with Nominatim(
                user_agent="Trial-AId",
                adapter_factory=AioHTTPAdapter
                ) as geolocator_nominatim:
                location = await geolocator_nominatim.geocode(address, timeout=timeout)
        if location is None:
            return np.nan, np.nan
        else:
            return location.latitude, location.longitude

    except geopy.exc.GeocoderQueryError as e:
        print(e)
        async with Nominatim(
                user_agent="Trial-AId",
                adapter_factory=AioHTTPAdapter
                ) as geolocator_nominatim:
                location = await geolocator_nominatim.geocode(address, timeout=timeout)
        if location is None:
            return np.nan, np.nan
        else:
            return location.latitude, location.longitude

    except geopy.exc.GeocoderRateLimited as e:
        print(e)
        sleep(10)
        geocode_async(address, timeout=timeout+10)

    except geopy.exc.GeocoderTimedOut as e:
        print(e)
        sleep(10)
        geocode_async(address, timeout=timeout+10)

    except ImportError as e:
        print(e)
        sleep(10)
        geocode_async(address, timeout=timeout+10)


def geocode(address):

    geolocator_bing = Bing(api_key="Amh-_uUK56C9ZaUCI63lVDVyJLRQGGOmgeNWVPrO0gu4YufXeCaNmOzOqmsZA-Vx")
    geolocator_nominatim = Nominatim(user_agent="Trial-AId")

    try:
        location = geolocator_bing.geocode(address)
        if location is None:
            location = geolocator_nominatim.geocode(address)
            if location is None:
                # sleep(1)
                return np.nan, np.nan
            else:
                # sleep(1)
                return location.latitude, location.longitude
        else:
            # sleep(1)
            return location.latitude, location.longitude

    except geopy.adapters.AdapterHTTPError as e:
        print(e)
        location = geolocator_nominatim.geocode(address)
        if location is None:
            # sleep(1)
            return np.nan, np.nan
        else:
            # sleep(1)
            return location.latitude, location.longitude

    except geopy.exc.GeocoderQueryError as e:
        print(e)
        location = geolocator_nominatim.geocode(address)
        if location is None:
            # sleep(1)
            return np.nan, np.nan
        else:
            # sleep(1)
            return location.latitude, location.longitude

    except geopy.exc.GeocoderTimedOut as e:
        print(e)
        geocode(address)


def get_dataframe(key):
    search = query(key)
    df = pd.DataFrame.from_records(search[1:], columns=search[0])
    df = df[df.OverallStatus != "Completed"]
    df = df[df.OverallStatus != "Terminated"]
    df = df[df.LocationFacility != ""]
    df["Link"] = df["NCTId"].apply(lambda x: "https://clinicaltrials.gov/ct2/show/" + x)
    df.drop(columns=["Rank", "NCTId"], inplace=True)
    return df


def get_coordinates(df):
    df["Latitude"] = df["LocationFacility"].apply(lambda x: geocode(x)[0])
    df["Longitude"] = df["LocationFacility"].apply(lambda x: geocode(x)[1])
    return df


async def get_coordinates_async(df):
    tasks = []
    for el in df["LocationFacility"].values:
        task = asyncio.ensure_future(geocode_async(el))
        tasks.append(task)
    coordinates = await asyncio.gather(*tasks, return_exceptions=True)
    print(coordinates)
    df["Latitude"] = [el[0] if el is not None else np.nan for el in coordinates]
    df["Longitude"] = [el[1] if el is not None else np.nan for el in coordinates]
    return df


def clean_coordinates(df):
    print(df)
    df[df["Longitude"].apply(lambda x: x is np.float64)]
    df[df["Latitude"].apply(lambda x: x is np.float64)]
    df.dropna(inplace=True)
    return df


def get_info(key):
    df = get_dataframe(key)
    df = get_coordinates(df)
    df = clean_coordinates(df)
    df.reset_index(drop=True, inplace=True)
    return df


def get_info_async(key):
    df = get_dataframe(key)
    df = asyncio.run(get_coordinates_async(df))
    df = clean_coordinates(df)
    df.reset_index(drop=True, inplace=True)
    return df


sync = True
if __name__ == "__main__":
    t1 = time()
    if sync:
        df = get_info("aspirin")
    else:
        df = get_info_async("aspirin")
    t2 = time() - t1
    print(df)
    print(t2)


# import json
# with open("seach_results_specific.json", "w") as f:
#     f.write(json.dumps(search))
