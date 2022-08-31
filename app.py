import io
import requests
import warnings

import optuna
import pandas as pd
import streamlit as st
import datarobot as dr
import matplotlib.pyplot as plt
from pandas import json_normalize
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings(
    "ignore", category=ExperimentalWarning, module="optuna.multi_objective"
)
warnings.filterwarnings("ignore", category=UserWarning)

api_key = st.sidebar.text_input("API_KEY", "")
datarobot_key = st.sidebar.text_input("DATAROBOT_KEY", "")
deployment_id1 = st.sidebar.text_input("DEPLOYMENT_ID1", "")
deployment_id2 = st.sidebar.text_input("DEPLOYMENT_ID2", "")
n_trials = st.sidebar.number_input(
    label="最適化回数", min_value=10, step=1, value=50, format="%d", max_value=100000
)
file_upload = st.sidebar.file_uploader("学習CSVをアップロード（探索区間設定のため）")

API_URL = "{url}/predApi/v1.0/deployments/{deployment_id}/predictions"
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800
MAX_WAIT = 60 * 60


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_predictions(data, deployment_id, url):
    """
    Make predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs-jp/users-guide/predictions/api/new-prediction-api.html

    Parameters
    ----------
    data : str
        Feature1,Feature2
        numeric_value,string
    deployment_id : str
        The ID of the deployment to make predictions with.

    Returns
    -------
    Response schema:
        https://app.datarobot.com/docs-jp/users-guide/predictions/api/new-prediction-api.html#response-schema

    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        "Content-Type": "text/plain; charset=UTF-8",
        "Authorization": "Bearer {}".format(api_key),
        "DataRobot-Key": datarobot_key,
    }

    url = API_URL.format(url=url, deployment_id=deployment_id)
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return a Python dict following the schema in the documentation
    # Return as is
    return predictions_response  # .json()


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = "{code} Error: {msg}".format(
            code=response.status_code, msg=response.text
        )
        raise DataRobotPredictionError(err_msg)


def get_prediction(df, deployment_id, url):
    buffer = io.BytesIO()
    wrapper = io.TextIOWrapper(buffer, encoding="utf-8", write_through=True)
    df.to_csv(wrapper)

    predictions_response = make_datarobot_deployment_predictions(
        buffer.getvalue(), deployment_id, url
    )

    if predictions_response.status_code != 200:
        try:
            message = predictions_response.json().get(
                "message", predictions_response.text
            )
            status_code = predictions_response.status_code
            reason = predictions_response.reason

            print(
                "Status: {status_code} {reason}. Message: {message}.".format(
                    message=message, status_code=status_code, reason=reason
                )
            )
        except ValueError:
            print("Prediction failed: {}".format(predictions_response.reason))
            predictions_response.raise_for_status()
    else:
        return json_normalize(predictions_response.json()["data"]).prediction.values


if "is_expanded" not in st.session_state:
    st.session_state["is_expanded"] = True
st.header("多目的(2)最適化")
if api_key and datarobot_key and deployment_id1 and deployment_id2 and file_upload:
    with st.expander("設定", expanded=st.session_state["is_expanded"]):
        dr.Client(token=api_key, endpoint="https://app.datarobot.com/api/v2")
        st.text(f"dr client version: {dr.__version__}")

        deployments = dr.Deployment.list()
        st.text(f"FYI: total number of deployments is: {len(deployments)}")

        deployment1 = dr.Deployment.get(deployment_id=deployment_id1)
        target1 = deployment1.model.get("target_name")
        url1 = deployment1.prediction_environment["name"]
        deployment2 = dr.Deployment.get(deployment_id=deployment_id2)
        target2 = deployment2.model.get("target_name")
        url2 = deployment2.prediction_environment["name"]

        col1, col2 = st.columns(2)
        col1.text(f"ターゲット 1: {target1}")
        col2.text(f"ターゲット 2: {target2}")

        df = pd.read_csv(file_upload)
        # ToDo: fix unnecessary ID problem
        df = df.drop(["ID", target1, target2], axis=1)
        df_stat = df.describe().T
        st.dataframe(df.describe(percentiles=[0.5]).T, width=1500)

        st.subheader("可変特徴量の設定")
        list_col = df.columns.tolist()
        list_flex = st.multiselect("可変特徴量を選ぶ：", list_col)
        val_flex = [None for _ in range(len(list_flex))]
        if len(list_flex) == 0:
            st.warning("一つ以上選んでください", icon="⚠️")

        with st.container():
            for i, col in enumerate(list_flex):
                min = float(df[col].min())
                max = float(df[col].max())
                val_flex[i] = st.slider(
                    col,
                    min,
                    max,
                    (min, max),
                    key=f"{col}_{i}",
                )

        list_fix = [col for col in list_col if col not in list_flex]
        val_fix = [None for _ in range(len(list_fix))]

        st.subheader("固定特徴量の設定")
        st.text("1行目のデータをデフォルトで入力しています")
        for i, col in enumerate(list_fix):
            with st.container():
                val_fix[i] = st.number_input(col, key=str(i), value=df.loc[0, col])

    # prepare for optimize
    _, col_button, _ = st.columns([3, 1.5, 2.5])
    with col_button:
        placeholder = st.empty()
    button_optimize = placeholder.button("最適化実行", disabled=True, key="btn_1")
    if len(list_flex) > 0:
        button_optimize = placeholder.button("最適化実行", key="btn_2")

    if button_optimize:
        st.session_state["is_expanded"] = False

        def objective(trail):
            df_target = pd.DataFrame(index=[0], columns=df.columns)
            for i, col in enumerate(list_flex):
                low = val_flex[i][0]  # df_stat.loc[col, "min"] * 0.8
                high = val_flex[i][1]  # df_stat.loc[col, "max"] * 1.2
                df_target[col] = trail.suggest_float(col, low, high, step=0.01)
            for col, val in zip(list_fix, val_fix):
                df_target[col] = val
            pred_1 = get_prediction(df_target, deployment_id1, url1)
            pred_2 = get_prediction(df_target, deployment_id2, url2)
            return pred_1, pred_2

        with st.spinner("最適中・・・"):
            sampler = optuna.samplers.NSGAIISampler()
            study = optuna.create_study(
                directions=["maximize", "maximize"], sampler=sampler
            )
            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    with st.expander("最適結果", expanded=True):
        # gather data
        trial_all = []
        trial_params = []
        for trial in study.get_trials():
            trial_all.append([trial.number, trial.values[0], trial.values[1]])
            trial_params.append(trial.params)
        trial_all = pd.DataFrame(trial_all, columns=["Iteration", target1, target2])
        trial_params = pd.DataFrame(trial_params)
        trial_all = pd.merge(trial_all, trial_params, left_index=True, right_index=True)

        trial_pareto = []
        for trial in study.best_trials:
            trial_pareto.append([trial.number, trial.values[0], trial.values[1]])
        trial_pareto = pd.DataFrame(
            trial_pareto, columns=["Iteration", target1, target2]
        )
        trial_pareto.sort_values(target1, inplace=True)
        # plot
        # import pdb
        # pdb.set_trace()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trial_all[target1], trial_all[target2], linestyle="", marker="*")
        ax.plot(trial_pareto[target1], trial_pareto[target2])
        ax.axis("equal")
        ax.set_xlabel(target1)
        ax.set_ylabel(target2)
        fig.tight_layout()
        # fig = optuna.visualization.plot_pareto_front(
        #     study, target_names=[target1, target2]
        # )
        st.pyplot(fig=fig)

        st.subheader("パラメータ")
        data = st.dataframe(trial_all)
