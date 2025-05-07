import os
import sys

import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np

# Get the parent directory (Project Root)
project_root = os.path.abspath(os.path.dirname(__file__))

sys.path.append(project_root)  # Add project root to Python's search path

def data_file(filename):
    return os.path.join(project_root, "data", filename)

# Annuity Function
def annuity(r, n):
    return r / (1 - 1 / (1 + r)**n) if r > 0 else 1 / n

def load_hydro_data(country, network):
    # Load and preprocess hydro inflow data
    inflow_ror_GWh_day = pd.read_csv(data_file(f"Hydro_Inflow_{country[:2]}.csv"))
    inflow_ror_GWh_day = inflow_ror_GWh_day[inflow_ror_GWh_day["Year"] == 2012]
    inflow_ror_GWh_day["date"] = pd.to_datetime(inflow_ror_GWh_day[["Year", "Month", "Day"]])
    inflow_ror_GWh_day.set_index("date", inplace=True)

    ## convert to MW (average over day)
    df_daily = inflow_ror_GWh_day[["Inflow [GWh]"]]
    df_daily["Inflow [MW]"] = df_daily["Inflow [GWh]"] * 1000 / 24  # MW average per hour
    ## Resample to hourly resolution (linear interpolation)
    inflow_ror_hourly = df_daily["Inflow [MW]"].resample("h").interpolate("linear")
    ## normalize
    p_nom_ror = inflow_ror_hourly.max()
    p_max_pu_ror = inflow_ror_hourly / p_nom_ror
    ## adapt to snapshot format
    p_max_pu_ror= p_max_pu_ror[:8760]
    p_max_pu_ror.index = network.snapshots

    return p_max_pu_ror


def load_time_series():
    # demand time series
    df_elec = pd.read_csv(data_file("electricity_demand.csv"), sep=';', index_col=0) # in MWh
    df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

    # capacity factors
    df_onshorewind = pd.read_csv(data_file('onshore_wind_1979-2017.csv'), sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

    df_offhorewind = pd.read_csv(data_file('offshore_wind_1979-2017.csv'), sep=';', index_col=0)
    df_offhorewind.index = pd.to_datetime(df_offhorewind.index)

    df_solar = pd.read_csv(data_file('pv_optimal.csv'), sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)
    
    return df_elec, df_onshorewind, df_offhorewind, df_solar


def load_technology_data(cost_year = 2025, technologies=["onwind", "offwind", "solar", "OCGT", 
                                                         "CCGT", "hydro", "ror", "coal", "lignite", 
                                                         "biomass CHP","battery storage","battery inverter",
                                                         "HVAC overhead", "central air-sourced heat pump"]):
    
    url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{cost_year}.csv"
    costs = pd.read_csv(url, index_col=[0, 1])

    # convert from kW to MW
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")

    defaults = {
        "FOM": 0,
        "VOM": 0,
        "efficiency": 1,
        "fuel": 0,
        "investment": 0,
        "lifetime": 25,
        "CO2 intensity": 0,
        "discount rate": 0.07,
    }
    costs = costs.value.unstack().fillna(defaults)

    # gas, coal and lignate based on UBA data 2025 https://doi.org/10.60810/openumwelt-7844
    costs.at["gas", "CO2 intensity"] = 0.392
    costs.at["lignite", "CO2 intensity"] = 0.86
    costs.at["coal", "CO2 intensity"] =1.119
    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

    costs["marginal_cost"] = costs["VOM"] + (costs["fuel"] / costs["efficiency"])
    annuity_results = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
    costs["capital_cost"] = (annuity_results + (costs["FOM"] / 100)) * costs["investment"]

    costs = costs.loc[costs.index.get_level_values(0).isin(technologies)]

    return costs


def build_network(solve=True, year =2015, carriers=["onwind","offwind","solar","OCGT","CCGT","hydro","ror","coal","lignite","biomass CHP",
                                                    #"battery storage",
                                                    ],
                country = "DEU", countries=["DEU"], coordinates={'DEU': (51.15, 10.45)}):
    
    # 1. Create the network
    network = pypsa.Network()
    snapshots = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    if len(snapshots) == 8784:
        # Create a mask to exclude February 29
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]
    network.set_snapshots(snapshots)

    # 2. Select country and Add bus
    network.add("Bus",
        f"{country}_elec",
        y = coordinates[country][0],
        x = coordinates[country][1],
        carrier="AC")
    

    # 3. Load time series and technology data
    df_elec, df_onshorewind, df_offhorewind, df_solar = load_time_series()
    costs = load_technology_data()
    p_max_pu_ror = load_hydro_data(country, network)

    # 4. Add demand as a load to the bus
    network.add("Load",
                name=f"{country}_load",
                bus=f"{country}_elec",
                p_set=df_elec[country].values)
    
    # 5. Adding carriers
    network.add(
        "Carrier",
        carriers,
        co2_emissions=[costs.at[c, "CO2 intensity"] for c in carriers],
    )

    # 6. Calculating capacity factor data
    CF_onwind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_offwind = df_offhorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]

    # 7. Adding Generators
    # Add "Dispatchable Generators" - limits related to resource scarcity
    network.add(
                "Generator",
                f"coal",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="coal",
                capital_cost=costs.at["coal", "capital_cost"],
                marginal_cost=costs.at["coal", "marginal_cost"],
                efficiency=costs.at["coal", "efficiency"],
                p_min_pu = 0.33,
                p_nom_max=26000,
                overwrite=True)

    network.add(
            "Generator",
            f"lignite",
            bus=f"{country}_elec",
            p_nom_extendable=True,  
            carrier="lignite",
            p_nom_max=21000,
            capital_cost=costs.at["lignite", "capital_cost"],
            marginal_cost=costs.at["lignite", "marginal_cost"],
            efficiency=costs.at["lignite", "efficiency"],
            p_min_pu = 0.4,
            overwrite=True)

    network.add(
            "Generator",
            f"biomass CHP",
            bus=f"{country}_elec",
            p_nom_extendable=True,
            carrier="biomass CHP",
            p_nom_max = 10000, # maximum capacity can be limited due to environmental constraints
            capital_cost=costs.at["biomass CHP", "capital_cost"],
            marginal_cost=costs.at["biomass CHP", "marginal_cost"],
            efficiency=costs.at["biomass CHP", "efficiency"],
            p_min_pu = 0.33,
            overwrite=True)

    # Peakers
    # add OCGT (Open Cycle Gas Turbine) generator - limits related to resource scarcity 
    network.add("Generator",
                f"OCGT",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="OCGT",
                p_nom_max=31000, 
                capital_cost = costs.at["OCGT", "capital_cost"],
                marginal_cost = costs.at["OCGT", "marginal_cost"],
                p_min_pu = 0.2,
                overwrite=True)
    

    # add ror hydro
    network.add(
                "Generator",
                name=f"ror",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="ror",
                p_max_pu=p_max_pu_ror.loc[network.snapshots],  
                capital_cost=costs.at["ror", "capital_cost"],       
                marginal_cost=costs.at["ror", "marginal_cost"],
                p_nom_max=6000,
                overwrite=True      
            )

    ## Variables - limits optional, in order to make the system feasible
    # add onshore wind generator 
    network.add("Generator",
                f"onwind",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="onwind",
                #p_nom_max=180000, # maximum capacity can be limited due to environmental constraints
                capital_cost = costs.at["onwind", "capital_cost"],
                marginal_cost = costs.at["onwind", "marginal_cost"],
                p_max_pu = CF_onwind.values,
                overwrite=True)

    # add offshore wind generator
    network.add("Generator",
                f"offwind",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="offwind",
                #p_nom_max=75000, # maximum capacity can be limited due to environmental constraints
                capital_cost = costs.at["offwind", "capital_cost"],
                marginal_cost = costs.at["offwind", "marginal_cost"],
                p_max_pu = CF_offwind.values,
                overwrite=True)

    # add solar
    network.add("Generator",
                f"solar",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="solar",
                #p_nom_max=385000, # maximum capacity can be limited due to environmental constraints
                capital_cost = costs.at["solar", "capital_cost"],
                marginal_cost = costs.at["solar", "marginal_cost"],
                p_max_pu = CF_solar.values,
                overwrite=True)

    if solve:
        network.optimize(solver_name="highs")

    return network

def get_costs():
    return global_costs.copy()

# This ensures the script doesn't run when imported
if __name__ == "__main__":
    build_network()