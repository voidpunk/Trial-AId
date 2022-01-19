import streamlit as st
import plotly.express as px
import pydeck as pdk
import pandas as pd
from model import query, get_info_n_pred
from infos import get_info_async, gather_data, TELEGRAM_API_KEY
from time import time
from PIL import Image


def textlayer_cleaner(df):
    df.sort_values(by=["Latitude", "Longitude"], inplace=True)
    for x in range(0, len(df)-1):
        if df.loc[x, "Latitude"] == df.loc[x+1, "Latitude"] and df.loc[x, "Longitude"] == df.loc[x+1, "Longitude"]:
            df.loc[x, "LocationFacility"] = ""
            df.loc[x+1, "LocationFacility"] = ""
    return df


def infos_cleaner(infos):
    if infos["first_approval"] is None:
        infos["first_approval"] = "-"
    if infos["indication_class"] is None:
        infos["indication_class"] = "-"
    if infos["natural_product"] == 1:
        infos["natural_product"] = "Yes"
    elif infos["natural_product"] == 0:
        infos["natural_product"] = "No"
    if infos["oral"] == 1:
        infos["oral"] = "Yes"
    elif infos["oral"] == 0:
        infos["oral"] = "No"
    if infos["parenteral"] == 1:
        infos["parenteral"] = "Yes"
    elif infos["parenteral"] == 0:
        infos["parenteral"] = "No"
    if infos["topical"] == 1:
        infos["topical"] = "Yes"
    elif infos["topical"] == 0:
        infos["topical"] = "No"
    infos["pref_name"] = infos["pref_name"].capitalize()
    infos["indication_class"] = infos["indication_class"].replace(";", ",")
    return infos


def model_section(molecule):

    info_n_pred = get_info_n_pred(molecule)
    infos = infos_cleaner(info_n_pred["infos"])
    graph = info_n_pred["graph"]
    clintox = info_n_pred["clintox_pred"]
    sider = info_n_pred["sider_pred"]
    bbbp = info_n_pred["bbbp_pred"]
    graph.visualize(save_file="graph.png",)
    product_issues = sider.pop("Product issues")
    infections_infestations = sider.pop("Infections & infestations")

    st.write("----------------------------------------------------------------")

    col1, col3 = st.columns(2)
    col1.write(
        f"""
        <div style="display: flex;">
            <div style="flex: 50%;">
                <p><b>Data from database:</b></p>
                <p>Name:</p>
                <p>First approval:</p>
                <p>Oral:</p>
                <p>Parenteral:</p>
                <p>Topical:</p>
                <p>Natural product:</p>
                <p>Indication class:</p>
            </div>
            <div style="flex: 50%;">
                <p><b>Values</b></p>
                <p><code>{infos["pref_name"]}</code></p>
                <p><code>{infos["first_approval"]}</code></p>
                <p><code>{infos["oral"]}</code></p>
                <p><code>{infos["parenteral"]}</code></p>
                <p><code>{infos["topical"]}</code></p>
                <p><code>{infos["natural_product"]}</code></p>
                <p><code>{infos["indication_class"]}</code></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col3.image(
        "graph.png",
        caption="Real time computer-reconstructed molecule shape",
        use_column_width=True
        )


    st.write("----------------------------------------------------------------")

    col1, col_empty, col2 = st.columns((5, 1, 3))
    col1.subheader("FDA approval")
    col1.write(
        """
        This measure estimates the likelihood of the drug to be approved by the FDA.
        """
    )
    # col1.write("<br>", unsafe_allow_html=True)
    if clintox["FDA approval"] < 50:
        col1.error("⚠️ Probably it won't be approved by FDA!")
    else:
        col1.success("✅ Probably it will be approved by FDA!")
    with col1.expander("More info:"):
        st.write(
            """
            <p style="text-align: justify;">
            FDA approval of a drug means that data on the drug's effects have been reviewed by CDER, and the drug is 
            determined to provide benefits that outweigh its known and potential risks for the intended population.
            <br><br>
            At the moment, the model precision for this measeure evaluate as >90%.
            </p>
            """,
            unsafe_allow_html=True
        )
    col2.plotly_chart(
        px.bar(
            pd.DataFrame({
                "y": clintox["FDA approval"],
                "": 0
                },
                index=[""]
            ),
            y="y",
            color_discrete_sequence=["#23bb52"],
            width=250,
            labels={
                "y": "Probability (%)",
                "index": "FDA approval",
                },
        ).update_layout(
            yaxis=dict(range=[0, 100])
            )
    )

    st.write("----------------------------------------------------------------")

    col1, col_empty, col2 = st.columns((5, 1, 3))
    col1.subheader("Toxicity score")
    col1.write(
        """
        This measure estimates the likelihood of the drug to be toxic.
        """
    )
    # col1.write("<br>", unsafe_allow_html=True)
    if clintox["toxicity"] > 50:
        col1.error("⚠️ Probably the molecule is toxic!")
    else:
        col1.success("✅ Probably the molecule isn't toxic!")
    with col1.expander("More info:"):
        st.write(
            """
            <p style="text-align: justify;">
            Drug toxicity can be defined as a diverse array of adverse effects which are brought about through drug 
            use at either therapeutic or non-therapeutic doses.
            <br><br>
            At the moment, the model precision for this measeure evaluate as >90%.
            </p>
            """,
            unsafe_allow_html=True
        )
    col2.plotly_chart(
        px.bar(
            pd.DataFrame({
                "y": clintox["toxicity"],
                "": 0
                },
                index=[""]
            ),
            y="y",
            color_discrete_sequence=["#ff4b4b"],
            width=250,
            labels={
                "y": "Probability (%)",
                "index": "Toxicity",
                },
        ).update_layout(
            yaxis=dict(range=[0, 100])
            )
    )

    st.write("----------------------------------------------------------------")

    st.subheader("Side effects")
    st.write(
        """
        This measures estimate the likelihood of the drug to cause different types of side effects on different body 
        systems or of different kinds.
        """
    )
    with st.expander("More info:"):
        st.write(
            """
            <p style="text-align: justify;">
            A side effect is usually regarded as an undesirable secondary effect which occurs in addition to the 
            desired therapeutic effect of a drug or medication. Side effects may vary for each individual depending 
            on the person's disease state, age, weight, gender, ethnicity and general health.
            <br><br>
            Side effects can occur when commencing, decreasing/increasing dosages, or ending a drug or medication 
            regimen. Side effects may also lead to non-compliance with prescribed treatment. When side effects of a 
            drug or medication are severe, the dosage may be adjusted or a second medication may be prescribed. 
            Lifestyle or dietary changes may also help to minimize side effects.
            <br><br>
            At the moment, the model precision for this measeure evaluate as ~70%.
            </p>
            """,
            unsafe_allow_html=True
        )
    sider_df = pd.DataFrame.from_dict(
                sider,
                orient="index",
                columns=["score"]
            )
    sider_df.sort_values(by="score", inplace=True)
    st.plotly_chart(
        px.bar(
            sider_df,
            y=sider_df.index,
            x="score",
            orientation="h",
            color="score",
            height=600,
            labels={
                "index": "Side effects",
                "score": "Prediction (%)",
                },
            title="Probability",
        ).update_layout(
            xaxis=dict(range=[0, 100])
            ),
        use_container_width=True
    )
    st.write(
        """
        <p style="text-align: justify;">
        The <b>probability</b> measures the likelihood (as percentage) of the drug's side effects to affect a specific 
        system of the body or to represent a specific kind of problem (indicated by the main medical specialty involved).
        <br><br>
        The <b>distribution</b> of probability shows how the adverse effects are distributed among the different body 
        systems or medical specialties, to give a better understanding of the overall adverse effects' distribution.
        """,
        unsafe_allow_html=True
        )
    st.plotly_chart(
        px.pie(
            sider_df,
            values="score",
            names=sider_df.index,
            height=500,
            labels={
                "index": "Side effects",
                "score": "Prediction (%)",
                },
            title="Distribution",
        ),
        use_container_width=True
    )

    st.write("----------------------------------------------------------------")

    col1, col_empty, col2 = st.columns((5, 1, 4))
    col1.subheader("BBB penetration")
    col1.write(
        """
        This measure estimates the likelihood of the molecule to penetrate the blood-brain barrier.
        """
    )
    # col2.write("<br><br>", unsafe_allow_html=True)
    with col1.expander("More info:"):
        st.write(
            """
            <p style="text-align: justify;">
            The blood-brain barrier (BBB) prevents entry into the brain of most drugs from the blood. The presence of 
            the BBB, is a double-edged sword: it prevents the drug from entering the brain, thus protecting it from 
            toxic substances, but it also makes difficult the development of new treatments of brain diseases, since 
            they have to cross the BBB to reach and treat the brain.
            <br><br>
            At the moment, the model precision for this measeure evaluate as >90%.
            </p>
            """,
            unsafe_allow_html=True
        )
    if bbbp["BBB penetration"]:
        col2.error("⚠️ Probably it will pass the BBB!")
    else:
        col2.success("✅ Probably it won't pass the BBB!")

    st.write("----------------------------------------------------------------")

    col1, col_empty, col2 = st.columns((5, 1, 3))
    col1.subheader("Product issues")
    col1.write(
        """
        This measure estimates the likelihood of the molecule to be involved in post-marketing product issues.
        """
    )
    col1.write("<br>", unsafe_allow_html=True)
    if product_issues > 50:
        col1.error("⚠️ Probably it will give product issues!")
    else:
        col1.success("✅ Probably it won't give product issues!")
    with col1.expander("More info:"):
        st.write(
            """
            <p style="text-align: justify;">
            Product issues concern the quality, authenticity,or safety of any medication. Problems with product quality 
            may occur during manufacturing, shipping, or storage. They include: counterfeit product, product contamination, 
            poor packaging or product mix-up, questionable stability, labeling concerns.
            <br><br>
            At the moment, the model precision for this measeure evaluate as ~70%.
            </p>
            """,
            unsafe_allow_html=True
        )
    col2.plotly_chart(
    px.bar(
        pd.DataFrame({
            "y": product_issues,
            "": 0
            },
            index=[""]
        ),
        y="y",
        color_discrete_sequence=["#ff4b4b"],
        width=250,
        labels={
            "y": "Probability %",
            "index": "Product issues",
            }
    ).update_layout(
        yaxis=dict(range=[0, 100])
        )
)

    st.write("----------------------------------------------------------------")


def info_section(key):
    global data_collection, TELEGRAM_API_KEY

    st.header("Infos")
    st.write(
        """
        <p style="text-align: justify;">
        Here you can take a look at the available trials for the searched drug which are recruting or will recruting 
        around the globe.
        <br>
        Use the mouse wheel on the map to zoom in and out, and hover the cursor over red dots to see the name of the 
        location. Below the map you can see the trials' information. Click on the title to go to the official 
        ClinicalTrials.gov website and see all the details about the trial.
        </p>
        """,
        unsafe_allow_html=True
    )
    df, df_complete = get_info_async(key)
    df = textlayer_cleaner(df)
    # df_red = df[df["OverallStatus"] == "Recruiting"]
    # df_yellow = df[df["OverallStatus"] == "Not yet recruiting"]

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            # map_style="mapbox://styles/mapbox/dark-v10",
            # initial_view_state=pdk.ViewState(
            #     pitch=45
            # ),
            tooltip={"text": "{LocationFacility}"},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    # data=df_red,
                    data=df[["LocationFacility", "Latitude", "Longitude"]],
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
        )
    )

    # with open(f"trials/{key}.csv", "w") as f:
    #     df.drop(["Rank"], axis=1).to_csv(f, index=False)
    col_e1, col, col_e2 = st.columns((1, 4, 1))
    csv = save_df(df_complete)
    download = col.download_button(
        "Download trials as CSV (compatible with Excel & LibreOffice)",
        data=csv,
        file_name=f"{key}.csv",
        mime="csv"
        )
    if data_collection and download:
        gather_data(key, TELEGRAM_API_KEY, download)


    st.write("----------------------------------------------------------------")

    # Show user table
    columns = st.columns((1, 1, 3))
    fields = ["Title", "Status", "Summary"]
    for col, field_name in zip(columns, fields):
        # header
        col.write(field_name)
    for x, _ in enumerate(range(len(df))):
        col1, col2, col3 = st.columns((1, 1, 3))
        # clean text
        df.BriefTitle = df.BriefTitle.str.replace("/", " / ")
        df.BriefSummary = df.BriefSummary.str.replace(r"|", "\n")
        # columns
        col1.write(f"[{df.BriefTitle[x]}]({df.Link[x]})")
        col2.write(df["OverallStatus"][x])
        col2.write("Trial starts on:\n")
        col2.write(df["StartDate"][x])
        col3.write(f"""<p style="text-align: justify;">{df["BriefSummary"][x]}</p>""", unsafe_allow_html=True)
        st.write("------------------------------------------------------------")


@st.cache
def save_df(df):
    df = df.drop(["Rank"], axis=1).to_csv(index=False).encode('utf-8')
    return df


def intro():

    image = Image.open("./logo_transparent.png")
    st.image(image, use_column_width="auto")
    # st.title("Trial AId")
    st.write("")
    st.write(
        """
        <p style="text-align: justify;">
        Currently all the pieces of information provided to the subjects involved in clinical trials come from the 
        preclinical phase on in vitro and animal models. Often they are not enough detailed or accurate to reliably 
        inform the patients about all the possible outcomes of the experimental treatment they are going going to 
        receive.
        <br><br>
        This is where Trial AId comes: it is an AI-powered tool that aims at providing more detailed and reliable 
        information to the patient about the possible outcomes of the clinical <b>trial</b>, in order to <b>aid</b> 
        more informed decisions when choosing to participate to a trial. This is possible thanks to a deep learning 
        algorithm trained on hundreds of thousands of molecules, their properties and interactions.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.header("Model")


def about():

    st.header("About")
    st.write(
        """
        <p style="text-align: justify;">
        Trial AId is based on the powerful Torchdrug library (written upon PyTorch) and two enormous databases: 
        PubChem (110+ millions of compounds) and ChEMBL (2.1+ millions of compounds). 
        The clinical trials data is retrieved from ClinicalTrials.gov (400,000+ studies from 220 countries).
        <br>
        The model is pre-trained with unsupervised deep-learning on 250,000 molecules from the ZINC250k dataset.
        After that, it is specifically trained on thousands of molecules on 3 datasets: ClinTox, SIDER, and BBBP.
        <br>
        Currently the model has a 90% accuracy on the ClinTox dataset,a 70% accuracy on selected tasks of the SIDER 
        dataset (all the one shown), and a 90% accuracy on the BBBP dataset.
        <br><br>
        The next goals are to pre-train the model on 2 million molecules from the ZINC2M dataset, to improve the 
        overall performance of the model, and to train the side-effect prediction model jointly on the SIDER, OFFSIDES, 
        MEDEFFECT, and FAERS datasets to greatly improve the accuracy of this model.
        Currently I am working on elaborating the FAERS dataset, the biggest dataset of drug side-effects, provided by 
        the FDA.
        <br><br>
        </p>
        """,
        unsafe_allow_html=True
    )


def footer():

    st.write("Want to contribute? Great!<br>Check out the Github repository:", unsafe_allow_html=True)
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
    pubchempy_shield = "https://img.shields.io/badge/pubchempy-grey.svg?&style=for-the-badge"

    st.write("Made with:")
    # st.markdown(f"![]({torchdrug_base}{torchdrug_logo})")https://torchdrug.ai/
    st.write(
        """
        <div align="center">
            <a href="https://torchdrug.ai/" style="text-decoration: none;">
                <img src="{0}{1}">
            </a>
            <a href="https://github.com/mcs07/PubChemPy" style="text-decoration: none;">
                <img src="{2}">
            </a>
            <a href="https://github.com/voidpunk/pytrials" style="text-decoration: none;">
                <img src="{3}">
            </a>
            <a href="https://github.com/chembl/chembl_webresource_client" style="text-decoration: none;">
                <img src="{4}">
            </a>
        </div>
        """.format(torchdrug_base, torchdrug_logo, pubchempy_shield, pytrials_shield, chembl_shield),
        unsafe_allow_html=True
        )

    st.write("By:")
    st.write(
        """
        <div align="center">
            <a href="https://voidpunk.github.io/" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/Website-003971?style=for-the-badge&label=🌐&labelColor=3a3c40">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        <div align="right">
            <p>
                Trial-AId v0.3.0
                <br>
                Nil
                <i>&#64voidpunk</i>
                2022
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



def main():
    global data_collection, TELEGRAM_API_KEY
    intro()
    key = st.text_input(
        label="ENTER THE MOLECULE NAME:",
        # value="amoxicillin",
        placeholder="e.g. amoxicillin",
        # autocomplete="on",
        )
    col1, col2 = st.columns((3, 6))
    data_collection = col1.checkbox("Allow data collection")
    with col2.expander("Accepted formats:"):
        st.write(
            """
            <p>
            market name (e.g. "aspirin")
            <br>
            molecule name (e.g. "acetylsalicylic acid")
            <br>
            PubChem CID (e.g. "2244")
            <br>
            ChEMBL ID (e.g. "CHEMBL25")
            <br>
            SMILES (e.g. "CC(=O)Oc1ccccc1C(=O)O")
            <br>
            InChI key (e.g. "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
            <br>
            formula (e.g. "C9H8O4")
            <br>
            InChI (e.g. "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")
            <br>
            <br>
            N.B.:
            <br>
            1. searching an InChI takes a bit longer
            <br>
            2. InChI and InChI key are case sensitive
            <br>
            3. formula is not unique, so it is not as reliable as the other identifiers
            </p>
            """,
            unsafe_allow_html=True
        )
    if key != "":
        key = key.strip()
        if data_collection:
            gather_data(key, TELEGRAM_API_KEY)
        t0 = time()
        # query ChEMBL database
        with st.spinner("Searching through 110+ million compounds..."):
            molecule = query(key)
        # easter egg and dedication
        if key == "Charlie":
            st.info("Hello honey <3")
        # check the input and the presence of the molecule in the database
        elif molecule is None:
            st.warning(
                "No data available for this molecule, did you enter the correct name?"
            )
        else:
            # run the model section
            model_section(molecule)
            # run the info section
            with st.spinner("Travelling the world searching for your trials..."):
                info_section(key)
        t1 = time()
        # benchmark log
        print(f"Time: {t1 - t0}s")
    about()
    footer()



if __name__ == "__main__":
    main()