import streamlit as st
import model as mod
import plotly.express as px
import pandas as pd


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

col1, col2, = st.columns(2)

if key != "":
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
    #     'some_group': 'red',
    #     'some_other_group': 'green'
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

st.header("About")
st.write("Trial AId is based on the powerful Torchdrug library and the extensive ChEMBL database.")

