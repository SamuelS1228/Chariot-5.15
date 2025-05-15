import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(page_title="Warehouse Network Optimizer", layout="wide")

# ─────────────────────────── Session bootstrap ────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

def _num_input(scn, key, label, default, fmt="%.4f", **kwargs):
    """Helper that persists numeric inputs to the scenario dict."""
    scn.setdefault(key, default)
    scn[key] = st.number_input(label, value=scn[key], format=fmt,
                               key=f"{key}_{scn['_name']}", **kwargs)

# ─────────────────────────── Sidebar renderer ─────────────────────────────
def render_sidebar(scn):
    name = scn['_name']
    with st.sidebar:
        st.header(f'Inputs — {name}')

        # Demand upload ----------------------------------------------------
        with st.expander('🗂️ Demand & Candidate Files', expanded=True):
            up = st.file_uploader('Store demand CSV (Longitude, Latitude, DemandLbs)',
                                  key=f'dem_{name}')
            if up:
                scn['demand_file'] = up
            if 'demand_file' not in scn:
                st.info('Upload a demand file to begin.')
                return False  # skip rest of sidebar

            if st.checkbox('Preview demand file', key=f'pr_{name}'):
                st.dataframe(pd.read_csv(scn['demand_file']).head())

            # Candidate sites ---------------------------------------------
            st.markdown('**Candidate Warehouse Sites**')
            cand_up = st.file_uploader('CSV (lon,lat[,cost_per_sqft])',
                                       key=f'cand_{name}')
            if cand_up is not None:
                # New upload or cleared
                if cand_up:
                    scn['cand_file'] = cand_up
                else:
                    scn.pop('cand_file', None)
            scn['restrict_cand'] = st.checkbox('Restrict to candidate sites only',
                                               value=scn.get('restrict_cand', False),
                                               key=f'rc_{name}')

        # Cost parameters --------------------------------------------------
        with st.expander('💰 Cost Parameters', expanded=False):
            st.subheader('Transportation $ / lb‑minute')
            _num_input(scn, 'rate_out', 'Outbound', 0.02, '%.4f')
            _num_input(scn, 'in_rate', 'Inbound', 0.01, '%.4f')
            _num_input(scn, 'trans_rate', 'Transfer (RDC ➜ WH)', 0.015, '%.4f')

            st.subheader('Warehouse Cost Parameters')
            _num_input(scn, 'sqft_per_lb', 'Sq ft per lb', 0.02, '%.4f')
            _num_input(scn, 'cost_sqft', 'Default $/sq ft / yr', 6.0, '%.2f')
            _num_input(scn, 'fixed_wh_cost', 'Fixed $/warehouse', 250000.0, '%.0f',
                       step=50000.0)

        # Drive‑time toggle -----------------------------------------------
        with st.expander('🚚 Distance & Drive‑Time', expanded=False):
            scn['drive_times'] = st.checkbox('Use real drive times (OpenRouteService)',
                                             value=scn.get('drive_times', False),
                                             key=f'dt_{name}')
            if scn['drive_times']:
                scn['ors_key'] = st.text_input('OpenRouteService API key',
                                               value=scn.get('ors_key', ''),
                                               type='password',
                                               key=f'ors_{name}')

        # k selection ------------------------------------------------------
        with st.expander('🔢 Warehouse Count', expanded=False):
            scn['auto_k'] = st.checkbox('Optimize # warehouses',
                                        value=scn.get('auto_k', True),
                                        key=f'ak_{name}')
            if scn['auto_k']:
                scn['k_rng'] = st.slider('k range', 1, 10,
                                         scn.get('k_rng', (2, 5)),
                                         key=f'k_rng_{name}')
            else:
                _num_input(scn, 'k_fixed', '# warehouses', 3, '%.0f',
                           step=1, min_value=1, max_value=10)

        # Fixed & Inbound/Outbound sites -----------------------------------
        with st.expander('📍 Locations', expanded=False):
            st.subheader('Fixed Warehouses')
            fixed_text = st.text_area('lon,lat per line',
                                      value=scn.get('fixed_text', ''),
                                      key=f'fx_{name}', height=80)
            scn['fixed_text'] = fixed_text
            fixed_centers = []
            for ln in fixed_text.splitlines():
                try:
                    lon, lat = map(float, ln.split(','))
                    fixed_centers.append([lon, lat])
                except Exception:
                    continue
            scn['fixed_centers'] = fixed_centers

            st.subheader('Inbound Flow (supply points)')
            scn['inbound_on'] = st.checkbox('Include inbound flow',
                                            value=scn.get('inbound_on', False),
                                            key=f'inb_{name}')
            inbound_pts = []
            if scn['inbound_on']:
                sup_text = st.text_area('lon,lat,pct (0‑100) per line',
                                        value=scn.get('sup_text', ''),
                                        key=f'sup_{name}', height=100)
                scn['sup_text'] = sup_text
                for ln in sup_text.splitlines():
                    try:
                        lon, lat, pct = map(float, ln.split(','))
                        inbound_pts.append([lon, lat, pct / 100.0])
                    except Exception:
                        continue
            scn['inbound_pts'] = inbound_pts

        # RDC / SDC (single) ----------------------------------------------
        with st.expander('🏬 RDC / SDC', expanded=False):
            st.write("Enable a single RDC (redistribution) or SDC (serves customers & redistributes).")
            en = st.checkbox('Enable RDC / SDC', key=f'rdc_en_{name}',
                             value=scn.get('rdc_en', False))
            scn['rdc_en'] = en
            if en:
                lon = st.number_input('Longitude', format='%.6f',
                                      value=float(scn.get('rdc_lon', 0.0)),
                                      key=f'rdc_lon_{name}')
                lat = st.number_input('Latitude', format='%.6f',
                                      value=float(scn.get('rdc_lat', 0.0)),
                                      key=f'rdc_lat_{name}')
                typ = st.radio('Type', ['RDC', 'SDC'],
                               index=0 if scn.get('rdc_typ', 'RDC') == 'RDC' else 1,
                               key=f'rdc_typ_{name}', horizontal=True)
                scn['rdc_lon'] = lon
                scn['rdc_lat'] = lat
                scn['rdc_typ'] = typ
            _num_input(scn, 'rdc_sqft_per_lb', 'RDC Sq ft per lb',
                       scn.get('sqft_per_lb', 0.02))
            _num_input(scn, 'rdc_cost_sqft', 'RDC $/sq ft / yr',
                       scn.get('cost_sqft', 6.0), '%.2f')

        # Run button -------------------------------------------------------
        st.markdown('---')
        if st.button('🚀 Run solver', key=f'run_{name}'):
            st.session_state['run_target'] = name
    return True  # sidebar rendered OK

# ─────────────────────────── Main layout ────────────────────────────────
tab_names = list(st.session_state['scenarios'].keys()) + ['➕ New scenario']
tabs = st.tabs(tab_names)

# Existing scenarios ------------------------------------------------------
for i, t in enumerate(tabs[:-1]):
    name = tab_names[i]
    scn = st.session_state['scenarios'][name]
    scn['_name'] = name
    with t:
        if not render_sidebar(scn):
            continue  # demand file not uploaded yet

        # Determine k_vals inside main pane (after sidebar input)
        if scn.get('auto_k', True):
            k_vals = list(range(int(scn.get('k_rng', (2, 5))[0]),
                                int(scn.get('k_rng', (2, 5))[1]) + 1))
        else:
            k_vals = [int(scn.get('k_fixed', 3))]
        st.session_state['k_vals'] = k_vals

        if st.session_state.get('run_target') == name:
            with st.spinner('Running optimization…'):
                # Demand data
                df = pd.read_csv(scn['demand_file'])
                for col in ['Longitude', 'Latitude', 'DemandLbs']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Longitude', 'Latitude', 'DemandLbs'])

                # Candidate sites and costs --------------------------------
                candidate_sites = None
                candidate_costs = None
                if scn.get('cand_file'):
                    try:
                        cand_df = pd.read_csv(scn['cand_file'], header=None)
                        cand_df = cand_df.apply(pd.to_numeric, errors='coerce')
                        cand_df = cand_df.dropna(subset=[0, 1])
                        if scn.get('restrict_cand'):
                            candidate_sites = cand_df.iloc[:, :2].values.tolist()
                        if cand_df.shape[1] >= 3:
                            candidate_costs = { (round(row[0], 6), round(row[1], 6)): row[2]
                                                for row in cand_df.itertuples(index=False) }
                    except EmptyDataError:
                        # File was cleared or empty; ignore
                        cand_df = None
                        scn.pop('cand_file', None)

                # Build RDC list (single)
                rdc_list = []
                if scn.get('rdc_en'):
                    rdc_list.append({
                        'coords': [scn['rdc_lon'], scn['rdc_lat']],
                        'is_sdc': scn.get('rdc_typ', 'RDC') == 'SDC'
                    })

                res = optimize(
                    df,
                    k_vals,
                    scn['rate_out'],
                    scn['sqft_per_lb'],
                    scn['cost_sqft'],
                    scn['fixed_wh_cost'],
                    consider_inbound=scn['inbound_on'],
                    inbound_rate_min=scn['in_rate'],
                    inbound_pts=scn['inbound_pts'],
                    fixed_centers=scn['fixed_centers'],
                    rdc_list=rdc_list,
                    transfer_rate_min=scn['trans_rate'],
                    rdc_sqft_per_lb=scn['rdc_sqft_per_lb'],
                    rdc_cost_per_sqft=scn['rdc_cost_sqft'],
                    use_drive_times=scn['drive_times'],
                    ors_api_key=scn.get('ors_key', ''),
                    candidate_sites=candidate_sites,
                    restrict_cand=scn.get('restrict_cand', False),
                    candidate_costs=candidate_costs,
                )
            plot_network(res['assigned'], res['centers'])
            summary(
                res['assigned'], res['total_cost'], res['out_cost'], res['in_cost'],
                res['trans_cost'], res['wh_cost'], res['centers'], res['demand_per_wh'],
                scn['sqft_per_lb'], rdc_enabled=bool(res.get('rdc_list')),
                consider_inbound=scn['inbound_on'], show_transfer=res['trans_cost'] > 0,
            )

# New scenario tab --------------------------------------------------------
with tabs[-1]:
    new_name = st.text_input('Scenario name')
    if st.button('Create scenario'):
        if new_name and new_name not in st.session_state['scenarios']:
            st.session_state['scenarios'][new_name] = {}
            st.experimental_rerun()