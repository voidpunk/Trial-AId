import streamlit as st
import model as mod

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

    inchi = mod.query(key)
    graph = mod.construct(inchi)

    graph.visualize(
        save_file="graph.png",
        # figure_size=(2, 2),
        )
    # st.pyplot(graph.visualize())

    col2.image(
        "graph.png",
        width=300,
        # caption="Graph",
        use_column_width=True
        )

    model, task = mod.load()
    pred = mod.predict(graph, model, task)

    col1.write("")
    col1.write("FDA approval: {}%".format(pred["FDA approved"]))
    col1.write("Toxic: {}%".format(pred["toxic"]))

st.header("About")
st.write("Trial AId is based on the powerful Torchdrug library and the extensive ChEMBL database.")

