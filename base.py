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

def build_network(solve=True, year =2015, countries=["DEU"], coordinates={'DEU': (51.15, 10.45)}):
    global global_costs
    
    # Annuity Function
    def annuity(r, n):
        return r / (1 - 1 / (1 + r)**n) if r > 0 else 1 / n

    # Base Model
    # 1. Create the network
    
    snapshots = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="H")
    # 2. Set the modeling year
    if len(snapshots) == 8784:
        # Create a mask to exclude February 29
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]
    
    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # 3. Select country and Add bus
    country = "DEU"  # Germany
    network.add("Bus",
        f"{country}_elec",
        y = coordinates[country][0],
        x = coordinates[country][1],
        carrier="AC")
    

    # 5. Load demand time series


    df_elec = pd.read_csv(data_file("electricity_demand.csv"), sep=';', index_col=0) # in MWh
    df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime


    # 6. Add demand as a load to the bus
    network.add("Load",
                name=f"{country}_load",
                bus=f"{country}_elec",
                p_set=df_elec[country].values)

    # 7. Importing capacity factor data

    df_onshorewind = pd.read_csv(data_file('onshore_wind_1979-2017.csv'), sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

    df_offhorewind = pd.read_csv(data_file('offshore_wind_1979-2017.csv'), sep=';', index_col=0)
    df_offhorewind.index = pd.to_datetime(df_offhorewind.index)

    df_solar = pd.read_csv(data_file('pv_optimal.csv'), sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)

    # filter for capacity factor data for Germany
    CF_onwind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_offwind = df_offhorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]

    carriers = [
        "onwind",
        "offwind",
        "solar",
        "OCGT",
        "CCGT",
        "hydro",
        "ror",
        "coal",
        "lignite",
        "biomass CHP",
        "battery storage",
    ]

    # 8. Add hydro inflow data

    # Load and preprocess hydro inflow data
    inflow_ror_GWh_day = pd.read_csv(data_file(f"Hydro_Inflow_{country[:2]}.csv"))
    inflow_ror_GWh_day = inflow_ror_GWh_day[inflow_ror_GWh_day["Year"] == 2012]
    inflow_ror_GWh_day["date"] = pd.to_datetime(inflow_ror_GWh_day[["Year", "Month", "Day"]])
    inflow_ror_GWh_day.set_index("date", inplace=True)

    # convert to MW (average over day)
    df_daily = inflow_ror_GWh_day[["Inflow [GWh]"]]
    df_daily["Inflow [MW]"] = df_daily["Inflow [GWh]"] * 1000 / 24  # MW average per hour
    # Resample to hourly resolution (linear interpolation)
    inflow_ror_hourly = df_daily["Inflow [MW]"].resample("H").interpolate("linear")
    # normalize
    p_nom_ror = inflow_ror_hourly.max()
    p_max_pu_ror = inflow_ror_hourly / p_nom_ror
    # adapt to snapshot format

    p_max_pu_ror= p_max_pu_ror[:8760]
    p_max_pu_ror.index = network.snapshots
    p_max_pu_ror.head()

    # 9. Importing cost data
    cost_year = 2025
    url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{cost_year}.csv"
    costs = pd.read_csv(url, index_col=[0, 1])

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

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

    costs["marginal_cost"] = costs["VOM"] + (costs["fuel"] / costs["efficiency"])
    annuity_results = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
    costs["capital_cost"] = (annuity_results + (costs["FOM"] / 100)) * costs["investment"]

    costs = costs.loc[costs.index.get_level_values(0).isin(
        ["onwind",
        "offwind",
        "solar",
        "OCGT",
        "CCGT",
        "hydro",
        "ror",
        "coal",
        "lignite",
        "biomass CHP",
        "battery storage",
        "battery inverter"
    ])]
    global_costs = costs
    # 10. Adding carriers
    network.add(
        "Carrier",
        carriers,
        co2_emissions=[costs.at[c, "CO2 intensity"] for c in carriers],
    )

    # 11. Adding Generators
    # Add "Dispatchable Generators"
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
                overwrite=True)

    network.add(
            "Generator",
            f"lignite",
            bus=f"{country}_elec",
            p_nom_extendable=True,  
            carrier="lignite",
            p_nom_max=30000,
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
            p_nom_max = 5000, # maximum capacity can be limited due to environmental constraints
            capital_cost=costs.at["biomass CHP", "capital_cost"],
            marginal_cost=costs.at["biomass CHP", "marginal_cost"],
            efficiency=costs.at["biomass CHP", "efficiency"],
            p_min_pu = 0.33,
            overwrite=True)

    # Peakers
    # add OCGT (Open Cycle Gas Turbine) generator
    network.add("Generator",
                f"OCGT",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="OCGT",
                # p_nom_max=2000, # limit for now, let's assume this is for peakers
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
                overwrite=True      
            )

    ## Variables
    # add onshore wind generator
    network.add("Generator",
                f"onwind",
                bus=f"{country}_elec",
                p_nom_extendable=True,
                carrier="onwind",
                #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
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
                #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
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
                #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
                capital_cost = costs.at["solar", "capital_cost"],
                marginal_cost = costs.at["solar", "marginal_cost"],
                p_max_pu = CF_solar.values,
                overwrite=True)
    

    if len(countries) > 1:
        for country in countries[1:]:
            # add demand 
            network.add("Bus",
                        f"{country}_elec",
                        y = coordinates[country][0],
                        x = coordinates[country][1],
                        carrier="AC")
            network.add("Load", 
                        f"{country}_load",
                        bus=f"{country}_elec",
                        p_set=df_elec[country].values,
                        carrier = "AC")
            network.add(
                "Generator",
                f"{country} OCGT",
                bus=f"{country}_elec",
                carrier="OCGT",
                capital_cost=costs.at["OCGT", "capital_cost"],
                marginal_cost=costs.at["OCGT", "marginal_cost"],
                efficiency=costs.at["OCGT", "efficiency"],
                p_nom_extendable=True,
            )   
            CF_wind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
            network.add("Generator",
                f"{country}_onwind",
                bus=f"{country}_elec",
                carrier="onwind",
                capital_cost=costs.at["onwind", "capital_cost"],
                marginal_cost=costs.at["onwind", "marginal_cost"],
                p_max_pu=CF_wind.values,
                efficiency=costs.at["onwind", "efficiency"],
                p_nom_extendable=True,
            )
            CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
            network.add("Generator",
                f"{country}_solar",
                bus=f"{country}_elec",
                carrier="solar",
                capital_cost=costs.at["solar", "capital_cost"],
                marginal_cost=costs.at["solar", "marginal_cost"],
                p_max_pu=CF_solar.values,
                efficiency=costs.at["solar", "efficiency"],
                p_nom_extendable=True,
            )
            network.add(
                "Line",
                f"DEU-{country}",
                bus0="DEU_elec",
                bus1=f"{country}_elec",
                s_nom = 1000,
                x = 1,
                r = 1,
            )



    if solve:
        network.optimize(solver_name="highs")

    return network

def get_costs():
    return global_costs.copy()

# This ensures the script doesn't run when imported
if __name__ == "__main__":
    build_network()