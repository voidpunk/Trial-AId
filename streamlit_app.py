import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from model import query, get_info_n_pred
from infos import get_info_async
from multiprocessing import Process
from time import time


def textlayer_cleaner(df):
    df.sort_values(by=["Latitude", "Longitude"], inplace=True)
    for x in range(0, len(df)-1):
        if df.loc[x, "Latitude"] == df.loc[x+1, "Latitude"] and df.loc[x, "Longitude"] == df.loc[x+1, "Longitude"]:
            df.loc[x, "LocationFacility"] = ""
            df.loc[x+1, "LocationFacility"] = ""
    return df


def model_section(molecule):
    col1, col2, = st.columns(2)
    col1.write("")

    info_n_pred = get_info_n_pred(molecule)
    infos = info_n_pred["infos"]
    graph = info_n_pred["graph"]
    clintox_df = info_n_pred["clintox_pred"]
    sider_df = info_n_pred["sider_pred"]
    bbbp_df = info_n_pred["bbbp_pred"]

    graph.visualize(
        save_file="graph.png",
        # figure_size=(2, 2),
        )
    # st.pyplot(graph.visualize())

    col1.write(infos)
    col2.image(
        "graph.png",
        width=300,
        # caption="Graph",
        use_column_width=True
        )

    clintox_fig = px.bar(
        clintox_df, x="score", y=clintox_df.index,
        labels={"score": "Probability", "index": ""}, color="score",
        height=250
        )
    # color_discrete_map={
    #     "some_group": "red",
    #     "some_other_group": "green"
    # }
    st.plotly_chart(clintox_fig, use_container_width=True)

    sider_df.sort_values(by="score", inplace=True)
    sider_fig = px.bar(
        sider_df, y=sider_df.index, x=sider_df["score"],
        labels={"score": "Probability", "index": ""}, color="score",
        orientation="h", height=900,
        )
    # fig.update_xaxes(tickangle=60)
    st.plotly_chart(
        sider_fig,
        use_container_width=True,
    )

    bbbp_fig = px.bar(
        bbbp_df, x="score", y=bbbp_df.index,
        labels={"score": "filled = BBB penetration", "index": ""},
        height=250,
    )
    st.plotly_chart(
        bbbp_fig,
        use_container_width=True
    )


def info_section(key):

    df = get_info_async(key)
    # st.write(df)
    st.header("Infos")

    df = textlayer_cleaner(df)
    # df_red = df[df["OverallStatus"] == "Recruiting"]
    # df_yellow = df[df["OverallStatus"] == "Not yet recruiting"]
    df_pdk = df[["LocationFacility", "Latitude", "Longitude"]]
    # st.write(df_pdk)

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        # initial_view_state=pdk.ViewState(
        #     pitch=45
        # ),
        tooltip={"text": "{LocationFacility}"},
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                # data=df_red,
                data=df_pdk,
                get_position="[Longitude, Latitude]",
                get_color=[200, 30, 0, 160],
                pickable=True,
                radius_scale=5,
                radius_min_pixels=10,
                radius_max_pixels=500,
            ),
            # pdk.Layer(
            #     "ScatterplotLayer",
            #     data=df_yellow,
            #     get_position="[Longitude, Latitude]",
            #     get_color=[242, 245, 66, 50],
            #     radius_scale=5,
            #     radius_min_pixels=10,
            #     radius_max_pixels=1000,
            # ),
        ]
    ))

    # Show user table
    columns = st.columns((1, 1, 3))
    fields = ["Title", "Status", "Summary"]
    for col, field_name in zip(columns, fields):
        # header
        col.write(field_name)
    for x, _ in enumerate(range(len(df))):
        col1, col2, col3 = st.columns((1, 1, 3))
        col1.write(f"[{df.OfficialTitle[x]}]({df.Link[x]})")
        col2.write(df["OverallStatus"][x])
        col2.write("Trial starts on:\n")
        col2.write(df["StartDate"][x])
        col3.write(df["BriefSummary"][x])



st.title("Trial AId")
st.write(
    """
    Currently all the pieces of information provided to the subjects involved in clinical trials come from the
    preclinical phase on animal models. Often they are not enough detailed or accurate to reliably inform the patients
    about all the possible outcomes of the experimental treatment which the subject is going to receive.

    This is where Trial AId comes: it is an AI-powered tool that aims at providing more detailed information to the
    patient about the possible outcomes of the clinical trial, in order to ***aid*** more informed decisions when
    choosing to participate to a ***trial***. This is possible thanks to a deep learning algorithm trained on hundreds
    of thousands of molecular structures and their corresponding properties and interactions.
    """
)

st.header("Model")
key = st.text_input("Enter the molecule name:")

if key != "":
    t0 = time()
    # query ChEMBL database
    molecule = query(key)
    # easter egg and dedication
    if key == "Charlie":
        st.info("Hello honey <3")
    # check the input and the presence of the molecule in the database
    elif molecule is None:
        st.warning("No data available for this molecule, did you enter the correct name?")
    else:
        # run the model section
        model_section(molecule)
        # run the info section
        with st.spinner(text="Travelling the world searching for your trials..."):
            info_section(key)
    t1 = time()
    # benchmark log
    print(f"Time: {t1 - t0}s")

st.header("About")
st.write(
    """
    Trial AId is based on the powerful Torchdrug library (written upon PyTorch) and the extensive ChEMBL database,
    containing more than 2.1 milions of chemical compounds. The clinical trials data is retrieved from ClinicalTrials.gov,
    an NIH website.

    The model is pre-trained with unsupervised deep-learning on 250,000 molecules from the ZINC250k dataset.
    After which, it is trained on thousands of molecules on 3 datasets: ClinTox, SIDER, and BBBP.
    The model currently has a 90% accuracy on the ClinTox dataset,a 70% accuracy on selected tasks of the
    SIDER dataset (all the one shown), and a 90% accuracy on the BBBP dataset.

    The next goal is to pre-train the model on 2 million molecules from the ZINC2M dataset, to improve the overall
    performance of the model, and to train the side-effect prediction model jointly on the SIDER, OFFSIDES, MEDEFFECT,
    and FAERS datasets to greatly improve the accuracy of this model.
    """
)
st.write("Want to contribute? Great! Check out the Github repository:")
st.write(
    """
    <div align="center">
        <a href="https://github.com/voidpunk/Trial-AId">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
        </a>
    </div>
    """,
    unsafe_allow_html=True
    )

torchdrug_base = "https://img.shields.io/badge/torchdrug-grey.svg?&style=for-the-badge&logo=data:image/svg%2bxml;base64,"
torchdrug_logo = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1MDAiIGhlaWdodD0iNTAwIiB2aWV3Qm94PSIwIDAgNTAwIDUwMCI+CiAgPG1ldGFkYXRhPjw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+Cjx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMTQyIDc5LjE2MDkyNCwgMjAxNy8wNy8xMy0wMTowNjozOSAgICAgICAgIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIvPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgCjw/eHBhY2tldCBlbmQ9InciPz48L21ldGFkYXRhPgo8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiBub25lOwogICAgICAgIHN0cm9rZTogI2U1MjYxZjsKICAgICAgICBzdHJva2Utd2lkdGg6IDQ1cHg7CiAgICAgIH0KCiAgICAgIC5jbHMtMSwgLmNscy0yIHsKICAgICAgICBmaWxsLXJ1bGU6IGV2ZW5vZGQ7CiAgICAgIH0KCiAgICAgIC5jbHMtMiB7CiAgICAgICAgZmlsbDogIzJmMmY0MTsKICAgICAgfQogICAgPC9zdHlsZT4KICA8L2RlZnM+CiAgPHBhdGggaWQ9IuWkmui+ueW9ol8xIiBkYXRhLW5hbWU9IuWkmui+ueW9oiAxIiBjbGFzcz0iY2xzLTEiIGQ9Ik0yNTAuNSwyN0w0NDMuOTk0LDEzOC41djIyM0wyNTAuNSw0NzMsNTcuMDA2LDM2MS41di0yMjNaIi8+CiAgPHBhdGggaWQ9IkQiIGNsYXNzPSJjbHMtMiIgZD0iTTIxOS45LDE1MS4wNHEtMTIuNzQ0LjI4Mi0yMi44MiwwLjU2LTExLjc2LjI4Mi0yMy42Ni0uMTR0LTE4LjktLjd2NS42cTguNjc2LDAuMjgyLDEzLjMsMi4xYTEwLjEwNiwxMC4xMDYsMCwwLDEsNi4xNiw2LjcycTEuNTM2LDQuOSwxLjU0LDE1LjI2VjMxOS4zMnEwLDEwLjA4LTEuNTQsMTUuMTJhOS43MzksOS43MzksMCwwLDEtNi4wMiw2LjcycS00LjQ4MiwxLjY4LTEzLjQ0LDIuMjRWMzQ5cTctLjU1OCwxOC45LTAuN3QyNC4yMi0uMTRxOS4yNCwwLDIxLjcuNDJUMjM4LjUyLDM0OXEzNSwwLDYwLjA2LTEyLjZ0MzguMjItMzUuNDJxMTMuMTU4LTIyLjgxOCwxMy4xNi01My4zNCwwLTQ2Ljc1OC0yNi44OC03MS44MnQtODIuMDQtMjUuMDZRMjMyLjY0LDE1MC43NiwyMTkuOSwxNTEuMDRabTY5LjAyLDI3LjcycTE0LjU1NiwyMi45NjIsMTQuNTYsNzAsMCwzMC41MjItNS44OCw1MS44VDI3OC4xNCwzMzIuOXEtMTMuNTg0LDExLjA2NC0zNy4xLDExLjA2LTEyLjg4MiwwLTE2Ljk0LTQuNzZ0LTQuMDYtMTkuMzJ2LTE0MHEwLTE0LjU1NiwzLjkyLTE5LjMydDE2LjgtNC43NlEyNzQuMzYsMTU1LjgsMjg4LjkyLDE3OC43NloiLz4KPC9zdmc+Cg=="
chembl_shield = "https://img.shields.io/badge/chembl-grey.svg?&style=for-the-badge"
pytrials_shield = "https://img.shields.io/badge/pytrials-grey.svg?&style=for-the-badge"

st.write("Made with:")
# st.markdown(f"![]({torchdrug_base}{torchdrug_logo})")https://torchdrug.ai/
st.write(
    """
    <div align="center">
        <a href="https://torchdrug.ai/" style="text-decoration: none;">
            <img src="{}{}">
        </a>
        <a href="https://github.com/chembl/chembl_webresource_client" style="text-decoration: none;">
            <img src="{}">
        </a>
        <a href="https://github.com/voidpunk/pytrials" style="text-decoration: none;">
            <img src="{}">
        </a>
    </div>
    """.format(torchdrug_base, torchdrug_logo, chembl_shield, pytrials_shield),
    unsafe_allow_html=True
    )