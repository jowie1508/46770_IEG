import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color map for consistent plotting
color_map = {
    "solar": "#f9d002",
    "onwind": "#1f78b4",
    "offwind": "#6baed6",
    "OCGT": "#e31a1c",
    "CCGT": "#fb9a99",
    "coal": "#8c564b",
    "lignite": "#7f3b08",
    "hydro": "#3182bd",
    "ror": "#9ecae1",
    "biomass CHP": "#33a02c",
    "waste CHP": "#ff7f00",
    "central gas CHP": "#a6cee3",
    "battery storage": "#6a3d9a",
    "battery inverter": "#b2df8a"
}


def plot_dispatch_week(network, start, end, title):
    week = network.generators_t.p.loc[start:end]
    colors = [color_map.get(gen, "#999999") for gen in week.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    week.plot.area(ax=ax, linewidth=0, color=colors)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Dispatch (MW)")
    ax.set_xlabel("Date")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_generation_bar(network):
    generation_by_generator = (
        network.generators_t.p
        .mul(network.snapshot_weightings.generators, axis=0)
        .sum()
    )

    generation_by_carrier = (
        generation_by_generator
        .groupby(network.generators.carrier)
        .sum()
        .div(1e6)  # MWh to TWh
    ).sort_values(ascending=False)

    colors = [color_map.get(carrier, "#999999") for carrier in generation_by_carrier.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    generation_by_carrier.plot.bar(ax=ax, color=colors)
    ax.set_title("Annual Electricity Generation by Technology (2015)")
    ax.set_ylabel("Energy [TWh]")
    ax.set_xlabel("Technology")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_generation_pie(network):
    generation_by_generator = (
        network.generators_t.p
        .mul(network.snapshot_weightings.generators, axis=0)
        .sum()
    )

    generation_by_carrier = (
        generation_by_generator
        .groupby(network.generators.carrier)
        .sum()
        .div(1e6)  # MWh to TWh
    ).sort_values(ascending=False)

    labels = generation_by_carrier.index
    sizes = generation_by_carrier.values

    mask = sizes > 0
    labels = labels[mask]
    sizes = sizes[mask]
    colors_filtered = [color_map.get(label, "#999999") for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        colors=colors_filtered,
        wedgeprops=dict(width=0.4, edgecolor='white')
    )

    for i, (wedge, label, value) in enumerate(zip(wedges, labels, sizes)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        ha = 'left' if x > 0 else 'right'
        ax.annotate(
            f"{label}\n{value:.1f} TWh\n{value / sizes.sum() * 100:.1f}%",
            xy=(x * 0.75, y * 0.75),
            xytext=(x * 1.2, y * 1.2),
            ha=ha,
            va='center',
            arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.2'),
            fontsize=9
        )

    ax.set_title("Annual Electricity Generation by Technology (2015)", pad=20)
    plt.tight_layout()
    plt.show()


def plot_duration_curves(network):
    fig, ax = plt.subplots(figsize=(10, 4))

    for tech in network.generators.carrier.unique():
        gen_ids = network.generators.index[network.generators.carrier == tech]
        if gen_ids.empty:
            continue
        dispatch = network.generators_t.p[gen_ids].sum(axis=1)
        dispatch_sorted = dispatch.sort_values(ascending=False).reset_index(drop=True)
        ax.plot(dispatch_sorted, label=tech, color=color_map.get(tech, "#999999"))

    ax.set_title("Duration Curves by Technology")
    ax.set_ylabel("Dispatch (MW)")
    ax.set_xlabel("Hours (sorted)")
    ax.legend(title="Technology")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_capacity_factors(network):
    cf = (
        network.generators_t.p.sum()
        / (network.generators.p_nom_opt * 8760)
    )

    cf_by_carrier = (
        cf.groupby(network.generators.carrier)
        .mean()
        .sort_values(ascending=False)
    )

    colors = [color_map.get(carrier, "#999999") for carrier in cf_by_carrier.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    cf_by_carrier.plot.bar(ax=ax, color=colors)

    ax.set_title("Average Capacity Factor by Technology (2015)")
    ax.set_ylabel("Capacity Factor")
    ax.set_xlabel("Technology")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def print_summary(network):
    print("=" * 40)
    print(f"System Cost: {round(network.objective / 1e9, 2)} billion euros")
    print("=" * 40)

    print("\nOptimal Generator and Storage Capacities (GW):")
    gen_caps = network.generators.p_nom_opt.div(1e3)
    storage_caps = network.storage_units.p_nom_opt.div(1e3)
    print(pd.concat([gen_caps, storage_caps]))

    print("\nOptimal Annual Energy Generation (GWh):")
    gen_energy = network.generators_t.p.sum().div(1e6)
    storage_energy = network.storage_units_t.p.sum().div(1e6)
    print(pd.concat([gen_energy, storage_energy]))

    print("=" * 40)



def plot_interannual_wind_capacity(df_capacities_yearly):
    """
    Plot the interannual variation of wind capacity (onshore and offshore) over the years.
    """
    # Check if both onshore and offshore wind data are available
    required_techs = ['onwind', 'offwind']
    if all(tech in df_capacities_yearly.index for tech in required_techs):
        df_wind_plot = df_capacities_yearly.loc[required_techs].T
    # Ensure the index is numeric (years) for plotting if it isn't already
        df_wind_plot.index = pd.to_numeric(df_wind_plot.index)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot onshore wind
        ax.plot(df_wind_plot.index,           # x-values (years)
                df_wind_plot['onwind'],        # y-values (capacity)
                marker='o',                    # Use circles as markers (dots)
                linestyle='-',                 # Connect markers with a solid line
                label='Onshore Wind',
                color=color_map.get('onwind', 'blue')) # Get color from map or default

        # Plot offshore wind
        ax.plot(df_wind_plot.index,           # x-values (years)
                df_wind_plot['offwind'],       # y-values (capacity)
                marker='s',                    # Use squares as markers (optional, 'o' is fine too)
                linestyle='--',                # Connect markers with a dashed line (optional)
                label='Offshore Wind',
                color=color_map.get('offwind', 'cyan')) # Get color from map or default

        # --- Formatting ---
        ax.set_title('Optimal Wind Capacity Variation by Weather Year')
        ax.set_xlabel('Weather Year')
        ax.set_ylabel('Installed Capacity (GW)')

        # Ensure all years are shown as ticks on the x-axis
        ax.set_xticks(weather_years)
        ax.tick_params(axis='x', rotation=0) # Keep year labels horizontal

        ax.legend() # Show the legend
        ax.grid(True, linestyle='--', alpha=0.6) # Add a grid for readability
        plt.tight_layout() # Adjust plot to prevent labels overlapping
        plt.show()

