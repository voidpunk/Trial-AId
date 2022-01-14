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
    clintox_pred = info_n_pred["clintox_pred"]
    sider_pred = info_n_pred["sider_pred"]

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

    # col1.write(clintox_pred)
    clintox_df = pd.DataFrame.from_dict(clintox_pred, orient="index", columns=["score"])
    # col1.write(clintox_df)

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

    sider_df = pd.DataFrame.from_dict(sider_pred, orient="index", columns=["score"])
    sider_df.sort_values(by="score", inplace=True)
    # st.write(sider_df)
    fig = px.bar(
        sider_df, y=sider_df.index, x=sider_df["score"],
        labels={"score": "Probability", "index": ""}, color="score",
        orientation="h", height=900,
        )
    # fig.update_xaxes(tickangle=60)
    st.plotly_chart(
        fig,
        use_container_width=True,
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
    # check the input and the presence of the molecule in the database
    if molecule is None:
        st.warning("No data available for this molecule, did you enter the correct name?")
    # easter egg and dedication
    elif molecule == "Charlie":
        st.write("Hello honey <3")
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
st.write("""
    Trial AId is based on the powerful Torchdrug library (written upon PyTorch) and the extensive ChEMBL database,
    containing more than 2.1 milions of chemical compounds. The clinical trials data is retrieved from ClinicalTrials.gov,
    an NIH website.

    The model is pre-trained with unsupervised deep-learning on 250,000 molecules from the ZINC250k dataset.
    After which, it is trained on thousands of molecules on 3 datasets: ClinTox, SIDER, and BBBP.unsupervised
    The model currently has a 90% accuracy on the ClinTox dataset,a 70% accuracy on selected tasks of the
    SIDER dataset (all the one shown), and a 90% accuracy on the BBBP dataset.

    The next goal is to pre-train the model on 2 million molecules from the ZINC2M dataset, to improve the overall
    performance of the model, and to train the side-effect prediction model jointly on the SIDER, OFFSIDES, MEDEFFECT,
    and FAERS datasets to greatly improve the accuracy of this model.
    """
)

