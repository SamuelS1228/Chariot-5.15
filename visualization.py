
import streamlit as st
import pandas as pd
import pydeck as pdk

_PAL = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
    [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
]
def _color(i): return _PAL[i % len(_PAL)]

def plot_network(stores, centers):
    st.subheader("Network Map")
    cen_df = pd.DataFrame(centers, columns=['Longitude', 'Latitude'])
    edges = [{
        'f': [r.Longitude, r.Latitude],
        't': [cen_df.iloc[int(r.Warehouse)].Longitude, cen_df.iloc[int(r.Warehouse)].Latitude],
        'color': _color(int(r.Warehouse)) + [160],
    } for r in stores.itertuples()]
    edge_layer = pdk.Layer('LineLayer', data=edges,
                           get_source_position='f', get_target_position='t',
                           get_color='color', get_width=2)
    cen_df[['r', 'g', 'b']] = [_color(i) for i in range(len(cen_df))]
    wh_layer = pdk.Layer('ScatterplotLayer', data=cen_df,
                         get_position='[Longitude,Latitude]',
                         get_fill_color='[r,g,b]', get_radius=35000, opacity=0.9)
    store_layer = pdk.Layer('ScatterplotLayer', data=stores,
                            get_position='[Longitude,Latitude]',
                            get_fill_color='[0,128,255]', get_radius=12000, opacity=0.6)
    deck = pdk.Deck(layers=[edge_layer, store_layer, wh_layer],
                    initial_view_state=pdk.ViewState(latitude=39, longitude=-98, zoom=3.5),
                    map_style='mapbox://styles/mapbox/light-v10')
    st.pydeck_chart(deck)

def summary(stores, total, out_cost, in_cost, trans_cost, wh_cost,
            centers, demand, sqft_per_lb, rdc_enabled=False,
            consider_inbound=False, show_transfer=False):
    st.subheader("Cost Summary")
    st.metric("Total Annual Cost", f"${total:,.0f}")
    cols = st.columns(4 if (consider_inbound or show_transfer) else 2)
    i = 0
    cols[i].metric("Outbound", f"${out_cost:,.0f}"); i += 1
    if consider_inbound:
        cols[i].metric("Inbound", f"${in_cost:,.0f}"); i += 1
    if show_transfer:
        cols[i].metric("Transfers", f"${trans_cost:,.0f}"); i += 1
    cols[i].metric("Warehousing", f"${wh_cost:,.0f}")

    cen_df = pd.DataFrame(centers, columns=['Longitude', 'Latitude'])
    cen_df['DemandLbs'] = demand
    cen_df['SqFt'] = cen_df['DemandLbs'] * sqft_per_lb
    st.subheader("Warehouse Demand & Size")
    st.dataframe(cen_df[['DemandLbs', 'SqFt', 'Latitude', 'Longitude']].style.format(
        {'DemandLbs': '{:,}', 'SqFt': '{:,}'}))
