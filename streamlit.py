import streamlit as st
import model as mod
import infos as inf
import pandas as pd
import plotly.express as px
import pydeck as pdk
# from pydeck.types import String
from multiprocessing import Process


# @st.cache
def get_locations_data(key):
    df = inf.get_dataframe(key)
    return df


# @st.cache
def get_locations_coordinates(df):
    df = inf.get_coordinates(df)
    return df


def textlayer_cleaner(df):
    df.sort_values(by=["Latitude", "Longitude"], inplace=True)
    for x in range(0, len(df)-1):
        if df.loc[x, "Latitude"] == df.loc[x+1, "Latitude"] and df.loc[x, "Longitude"] == df.loc[x+1, "Longitude"]:
            df.loc[x, "LocationFacility"] = ""
            df.loc[x+1, "LocationFacility"] = ""
    return df


# @st.cache(suppress_st_warning=True)
def model(key):
    col1, col2, = st.columns(2)
    col1.write("")

    inchi, infos = mod.query(key)
    graph = mod.construct(inchi)

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

    clintox_model, clintox_task = mod.load_clintox()
    clintox_pred = mod.predict(graph, clintox_model, clintox_task, "clintox")
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

    sider_model, sider_task = mod.load_sider()
    sider_pred = mod.predict(graph, sider_model, sider_task, "sider")

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


def infos(key):
    df = get_locations_data(key)
    df.reset_index(inplace=True)

    st.header("Infos")

    df = get_locations_coordinates(df)
    df = textlayer_cleaner(df)
    # st.write(df)
    # df_red = df[df["OverallStatus"] == "Recruiting"]
    # df_yellow = df[df["OverallStatus"] == "Not yet recruiting"]

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
                data=df,
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
            # pdk.Layer(
            #     "TextLayer",
            #     df,
            #     pickable=True,
            #     get_position="[Longitude, Latitude]",
            #     get_text="LocationFacility",
            #     get_size=16,
            #     get_color=[0, 0, 0],
            #     get_angle=0,
            #     # Note that string constants in pydeck are explicitly passed as strings
            #     # This distinguishes them from columns in a data set
            #     get_text_anchor=String("middle"),
            #     get_alignment_baseline=String("center"),
            # )
        ]
    ))

    # st.dataframe(df)

    # Show user table
    columns = st.columns((1, 1, 3))
    fields = ["Title", "Status", "Start Date", "Summary", "Location", "Link"]
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
        # button_phold = col4.empty()  # create a placeholder
        # do_action = button_phold.button("Link", key=x)
        # if do_action:
        #         pass # do some action with a row's data
        #         button_phold.empty()  #  remove button


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
    p1 = Process(target=infos, args=(key,))
    p2 = Process(target=model, args=(key,))
    p1.start()
    p2.start()
    model(key)
    infos(key)

st.header("About")
st.write("Trial AId is based on the powerful Torchdrug library and the extensive ChEMBL database.")

