import pandas as pd
import numpy as np
import os
import h5py as h5
import sqlalchemy as sa
import tomllib
from scipy.fft import fft
from scipy.optimize import curve_fit

from pathlib import Path

def get_county_list():
    with get_db_engine().connect() as con:
        counties = pd.read_sql_query(
            sql="select geoid as countyfp from region.county order by geoid",
            con=con,
        ).squeeze(axis=1)
    return counties.tolist()

def get_db_engine():
    secret = tomllib.load(open("./secret.toml", "rb"))
    url = sa.engine.URL.create(
        drivername="postgresql+psycopg",
        database=secret["atlas"]["database"],
        host=secret["atlas"]["hostname"],
        username=secret["atlas"]["username"],
        password=secret["atlas"].get("password"),  # optional, if not specified will search for .pgpass
    )
    
    return sa.create_engine(url)

def get_rev_timeseries_wind(turbine_class, rev_index_wind, year=2012, hrs_per_yr=8760):
    path_rev_wind = Path("/projects/dwind/data/rev/") / f"rev_{turbine_class:s}_generation_{year:d}.h5"
    rev_index_wind = np.sort(np.array(rev_index_wind, dtype=int))

    with h5.File(path_rev_wind.resolve(), "r") as hf:
        slice_step, mod = divmod(hf["cf_profile"].shape[0], hrs_per_yr)
        if mod != 0:
            raise NotImplementedError()

        scale_factor = hf["cf_profile"].attrs.get("scale_factor", 1000.0)
        cf_profile = hf["cf_profile"][slice(None, None, slice_step), rev_index_wind].T

    cf_profile = np.divide(cf_profile, scale_factor, dtype=np.float64)

    cf_profile = pd.Series(
        list(cf_profile),
        index=pd.Index(rev_index_wind, name="rev_index_wind"),
        name="cf_profile_hourly",
    )
    return cf_profile

turbine_class_dict = {
    2.5: "res",
    5.0: "res",
    10.0: "res",
    20.0: "res",
    50.0: "com",
    100.0: "com",
    250.0: "mid",
    500.0: "mid",
    750.0: "mid",
    1000.0: "large",
    1500.0: "large"
}

county_list = get_county_list()
all_county_emissions = []

usecols = np.arange(1,23,1)
cambium = pd.read_excel('cambium_lrmer.xlsx', skiprows=348, usecols=usecols)
usecols = np.arange(4,23,1)
cambiumTOD = pd.read_excel('cambium_lrmer.xlsx', skiprows=26, usecols=usecols, skipfooter=9058)

rev_lkup = pd.read_csv("/projects/dwind/configs/rev/wind/lkup_rev_index_2012_to_2018.csv")

for countyfp in county_list:

    print(f"Processing county: {countyfp}")

    with get_db_engine().connect() as con:
        agentbtm = pd.read_sql_query(
            sql="""--sql
            select a.gid, r.run_id, countyfp, statefp, turbine_class, wind_aep_kwh, techpot_kw, r.capacity_kw, application, rev_gid_wind, rev_index_wind, geom 
            from dwind.result as r
            join dwind.agent as a on r.gid = a.gid
            join dwind.breakeven_threshold bt on bt.run_id = r.run_id and bt.capacity_kw = r.capacity_kw
            where
                r.run_id = 6
                and a.countyfp = %(countyfp)s
                and r.breakeven_cost > 0 --bt.breakeven_threshold
            """,
            con=con,
            params={"countyfp": countyfp},
        )

    if agentbtm.empty:
        print(f"  Skipping county: no agent data.")
        continue
    else:
        agentbtm = agentbtm.merge(rev_lkup, left_on="rev_index_wind", right_on="rev_index_wind_2018")
        agentbtm = agentbtm.drop(columns=["rev_index_wind", "rev_index_wind_2018"])
        agentbtm = agentbtm.rename(columns={"rev_index_wind_2012": "rev_index_wind"})
        agentbtm["turbine_class"] = agentbtm["capacity_kw"].map(turbine_class_dict)
        agent = agentbtm

    agent.loc[agent.application=="FOM, Utility","application"]="FOM"
    agent.loc[agent.application=="BTM, FOM, Utility","application"]="BTM, FOM"

    turbine_class_rev_index = agent.groupby("turbine_class")["rev_index_wind"].unique()

    turbine_class_rev_index = pd.concat(
        objs={
            turbine_class: get_rev_timeseries_wind(turbine_class, rev_indexes)
            for turbine_class, rev_indexes in turbine_class_rev_index.items()
        },
        names=["turbine_class"],
    )

    agent = agent.merge(
        turbine_class_rev_index,
        left_on=turbine_class_rev_index.index.names,
        right_index=True,
    )

    if "cf_profile_hourly" not in agent.columns:
        print(f"  Skipping county {countyfp}: no cf_profile_hourly column")
        continue

    agent["wind_generation_hourly"] = agent["cf_profile_hourly"] * (agent["capacity_kw"] )
    agent["aep_high_res"] = agent["wind_generation_hourly"].apply(np.sum, axis=0)
    agent["aep_diff"] = agent["aep_high_res"] - agent["wind_aep_kwh"]

    profile = agent.groupby("countyfp", as_index=False).agg(
    {
        "wind_generation_hourly": lambda x: np.sum(np.stack(x), axis=0),
    }
    )

    timeseries_result = profile["wind_generation_hourly"].map(np.sum).sum()
    normal_result=agent["wind_aep_kwh"].sum().squeeze()
    percent_diff = (timeseries_result - normal_result) / normal_result * 100

    fips = pd.read_csv('Cambium_FIPS.csv')
    fips['County FIPS'] = fips['County FIPS'].astype(str)
    fips['State FIPS'] = fips['State FIPS'].astype(str)
    fips['GEOID'] = '0' 
    for i in range(len(fips)):
        if len(fips.loc[i, 'County FIPS']) ==1:
            fips.loc[i, 'GEOID'] = fips.loc[i, 'State FIPS'] + '00' + fips.loc[i, 'County FIPS']
        if len(fips.loc[i, 'County FIPS']) ==2:
            fips.loc[i, 'GEOID'] = fips.loc[i, 'State FIPS'] + '0' + fips.loc[i, 'County FIPS']
        if len(fips['County FIPS'][i]) ==3:
            fips.loc[i, 'GEOID'] = fips.loc[i, 'State FIPS'] + fips.loc[i, 'County FIPS']
        if len(fips.loc[i, 'State FIPS']) ==1:
            fips.loc[i, 'GEOID'] = '0' + fips.loc[i, 'GEOID']
    fips['County'] = fips['County'] + ", " + fips['State']

    fips = fips[['Cambium GEA', 'GEOID']] #we care about cambium GEA and GEOID only i think

    btm = fips.rename(columns={"GEOID":"countyfp"}).merge(agent, how='right', on='countyfp')

    btm['hourly_emissions'] = btm.apply(
        lambda row: row['wind_generation_hourly'] * cambium[row['Cambium GEA']].values,
        axis=1)

    btm['annual_emissions'] = np.sum(np.stack(btm['hourly_emissions'].values), axis=1)

    hour_of_day = np.tile(np.arange(24), 365)  # 365 * 24 = 8760
    def average_by_hour(x):
        return pd.Series(x).groupby(hour_of_day).mean().values  # returns a 24-element array
    btm['TOD_generation'] = btm['wind_generation_hourly'].apply(average_by_hour)
    btm['TOD_emissions'] = btm.apply(
        lambda row: row['TOD_generation'] * cambiumTOD[row['Cambium GEA']].values,
        axis=1)

    county_emissions = btm.groupby("countyfp").agg({
        "annual_emissions": "sum",
        "Cambium GEA": "first"  # or "mode", "max", etc., depending on what you want
    }).reset_index()

    hours = np.arange(0, 24, 1) #tod
    hourly_lrmer = np.stack(county_emissions.apply(
        lambda row: cambiumTOD[row['Cambium GEA']].values,
        axis=1))

    hourly_county_gen = np.sum(np.stack(btm['TOD_generation'].values), axis=0)

    dates = pd.date_range(start='2035-01-01', periods=8760, freq='h')
    hourly_county_gen_formonthly = np.sum(np.stack(btm['wind_generation_hourly'].values), axis=0)
    hourly_series_gen = pd.Series(hourly_county_gen_formonthly, index=dates)
    monthly_county_gen = hourly_series_gen.resample('ME').sum()
    monthly_county_gen.index = monthly_county_gen.index.strftime('%b')  # e.g., Jan, Feb

    # --- monthly data ---
    hourly_lrmer_formonthly = np.stack(county_emissions.apply(
        lambda row: cambium[row['Cambium GEA']].values,
        axis=1))

    hourly_series = pd.Series(hourly_lrmer_formonthly[0], index=dates)
    monthly_lrmer = hourly_series.resample('ME').sum()
    monthly_lrmer.index = monthly_lrmer.index.strftime('%b')  # e.g., Jan, Feb

    def cosine_similarity(A, B):
        # The time-series data sets should be normalized.
        A_norm = (A - np.mean(A)) / np.std(A)
        B_norm = (B - np.mean(B)) / np.std(B)

        # Determining the dot product of the normalized time series data sets.
        dot_product = np.dot(A_norm, B_norm)

        # Determining the Euclidean norm for each normalized time-series data collection.
        norm_A = np.linalg.norm(A_norm)
        norm_B = np.linalg.norm(B_norm)

        # Calculate the cosine similarity of the normalized time series data 
        # using the dot product and Euclidean norms. setse-series data set
        cosine_sim = dot_product / (norm_A * norm_B)

        return cosine_sim
    
    def cosine_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(frequency * x + phase) + offset

    # --- Helper function to fit cosine and return parameters + fit curve ---
    def fit_cosine(x_data, y_data):
        N = len(y_data)
        yf = fft(y_data)
        xf = np.fft.fftfreq(N, x_data[1] - x_data[0])

        # Dominant frequency (ignoring DC at index 0)
        idx = np.argmax(np.abs(yf[1:N//2])) + 1
        estimated_frequency = xf[idx] * 2 * np.pi  # angular freq

        # Amplitude & offset estimates
        estimated_amplitude = (np.max(y_data) - np.min(y_data)) / 2
        estimated_offset = np.mean(y_data)

        # Initial guess: [A, ω, φ, offset]
        p0 = [estimated_amplitude, estimated_frequency, 0, estimated_offset]

        popt, _ = curve_fit(cosine_func, x_data, y_data, p0=p0)
        return popt, cosine_func(x_data, *popt)

    # --- Data sets ---
    x_data1 = hours
    y_data1 = hourly_county_gen

    x_data2 = hours
    y_data2 = hourly_lrmer[0]

    # --- Fit both datasets ---
    params1, fit_curve1 = fit_cosine(x_data1, y_data1)
    params2, fit_curve2 = fit_cosine(x_data2, y_data2)

    phase_gen = params1[2]   # phase for county generation fit
    phase_lrmer = params2[2] # phase for lrmer fit

    # --- Compute phase difference ---
    phase_diff_tod = phase_lrmer - phase_gen

        # --- Data sets ---
    x_data1 = np.arange(len(monthly_county_gen))
    y_data1 = monthly_county_gen

    x_data2 = np.arange(len(monthly_county_gen))
    y_data2 = monthly_lrmer

    # --- Fit both datasets ---
    params1, fit_curve1 = fit_cosine(x_data1, y_data1)
    params2, fit_curve2 = fit_cosine(x_data2, y_data2)

    phase_gen = params1[2]   # phase for county generation fit
    phase_lrmer = params2[2] # phase for lrmer fit

    # --- Compute phase difference ---
    phase_diff_month = phase_lrmer - phase_gen

    county_emissions['hour_cs'] = cosine_similarity(hourly_lrmer[0], hourly_county_gen)
    county_emissions['month_cs'] = cosine_similarity(monthly_lrmer, monthly_county_gen)
    county_emissions['countyfp'] = countyfp
    county_emissions['timeseries_normal_percentdiff'] = percent_diff
    county_emissions['phase_diff_tod'] = phase_diff_tod
    county_emissions['phase_diff_month'] = phase_diff_month
    all_county_emissions.append(county_emissions)

final_emissions = pd.concat(all_county_emissions, ignore_index=True)
final_emissions.to_csv("county_emissions_btm_2phasediffs.csv", index=False)