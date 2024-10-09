import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import numpy as np

from tqdm import tqdm
from typing import Dict, Set

warnings.filterwarnings("ignore")

cl = pd.read_parquet(r"columns.parquet")
cols = dict(zip(cl['columns'].values, cl['rus'].values))

@st.cache_data
def load_full_dataset(path: str = r"full_result_2024_08_18.parquet") -> pd.DataFrame:
    dataset = pd.read_parquet(path).drop_duplicates()
    cols_rm_fd = list(cols.keys())
    cols_rm_fd.remove("description")
    cols_rm_fd.remove("image")
    to_add_categorical = ['floors-total', 'built-year', 'ready-quarter', 'lift', 'parking', 'rooms', 'floor']
    categorical_vals = sorted(set([i for i in tqdm(cols_rm_fd, position=0, leave=True) if str(dataset[i].dtype) == 'object'] + to_add_categorical))
    print("categorical_vals: ", categorical_vals)
    categorical_vals = {i: set(dataset[i].values) for i in tqdm(categorical_vals, position=0, leave=True)}
    dataset.fillna({i: "нет информации" for i in categorical_vals}, inplace=True)
    for col in categorical_vals:
        dataset[col] = dataset[col].astype(str)

    float_vals = sorted(set(cols_rm_fd) - set(categorical_vals))
    dataset.fillna({i: -1 for i in float_vals}, inplace=True)
    print("float_vals: ", float_vals)
    float_vals = {i: [min(dataset[i].values), max(dataset[i].values)] for i in tqdm(float_vals, position=0, leave=True)}

    return dataset, categorical_vals, float_vals

def initialize_state():
    """
    Initializes all filters and counter in Streamlit Session State
    """
    for subkey in cols:
        if subkey not in st.session_state:
            st.session_state[subkey] = set()
            
    if 'last_chosen' not in st.session_state:
        st.session_state['last_chosen'] = None
    if 'automatic' not in st.session_state:
        st.session_state['automatic'] = False
    if "all_values_di" not in st.session_state:
        st.session_state['all_values_di'] = cols
    if "counter" not in st.session_state:
        st.session_state.counter = 0

def reset_state_callback():
    """
    Resets all filters and increments counter in Streamlit Session State
    """
    st.session_state.counter = 1 + st.session_state.counter

    for subkey in cols:
        st.session_state[subkey] = set()
    st.session_state['last_chosen'] = None
    st.session_state['automatic'] = False
    
def update_state(current_query: Dict[str, Set], categorical_vals):
    """
    Stores input dict of filters into Streamlit Session State.

    If one of the input filters is different from previous value in Session State, 
    rerun Streamlit to activate the filtering and plot updating with the new info in State.
    """
    rerun = False
    for subkey in cols:
        if subkey in current_query:
            if st.session_state[subkey] != current_query[subkey]:
                st.session_state[subkey] = current_query[subkey]
                rerun = True
                if subkey in categorical_vals:
                    st.session_state['last_chosen'] = subkey
    if rerun:
        st.rerun()

def get_ui(cols, dataset, categorical_vals, float_vals):
    """
    Get UI
    """
    st.header("Подбираем недвижимость")
    di_vals = st.session_state['all_values_di']

    print("Dataset shape: ", dataset.shape)
    for col in categorical_vals:
        if st.session_state[col]:
            print(col, "-->", st.session_state[col])
            dataset = dataset[dataset[col].isin(st.session_state[col])]
    for col in float_vals:
        if st.session_state[col]:
            print(col, "-->", st.session_state[col])
            dataset = dataset[dataset[col].between(*st.session_state[col])]
    if len(dataset) == 0:
        st.write(":red[Исковых предложений нет!]")
        return dict()
    geo_data = dataset[['location_latitude', 'location_longitude', 'building-name']] # location_address building-name
    geo_data['to_count'] = 1
    geo_data = geo_data.groupby(['location_latitude', 'location_longitude', 'building-name'], as_index=False).sum() # location_address building-name
    print("Dataset shape after: ", dataset.shape)
    
    opts = dict()
    new_di_vals = {col_val: sorted(set(dataset[col_val].values)) for col_val in categorical_vals}
    if st.session_state['last_chosen']:
        new_di_vals[st.session_state['last_chosen']] = di_vals[st.session_state['last_chosen']]
    st.session_state['all_values_di'] = new_di_vals
    
    new_di_vals_float = {i: [min(dataset[i].values), max(dataset[i].values)] for i in float_vals}
    
    st.subheader(":rainbow[Введите текстовый запрос если удобно]")
    text_input = st.text_input("Текстовый запрос", "")

    ################################# Параметры квартиры #################################
    st.subheader(":rainbow[Параметры квартиры]")
    cat_cols_house = ['floor', 'rooms'] # 'renovation', 'balcony', 'bathroom-unit', 
    cols_house = st.columns(len(cat_cols_house), gap='large')
    cnt = 0
    for key in cat_cols_house:
        ln = len(new_di_vals[key])
        chosen_vals = [val for val in st.session_state[key] if val in new_di_vals[key]]
        with cols_house[cnt]:
            options = st.multiselect(
                f"Выберите :green[{cols[key]}] :gray[(осталось вариантов {ln})]:",
                new_di_vals[key],
                chosen_vals,
                placeholder="Выберите опцию")
        opts.update({key: set(options)})
        cnt = cnt + 1
    #print('-='*20, list(float_vals.keys()))
    #if distMetro:
    #    opts.update({'price_value': [number_price1, number_price2]})
    
    ################################# Параметры квартиры - float + остатки #################################
    cols_house_fl1 = st.columns(4, gap='large')
    with cols_house_fl1[0]:
        vals = new_di_vals_float['price_value']
        number_price1 = st.number_input(":green[Стоимость квартиры], от:", min_value=vals[0],
                                 max_value=vals[1], step=100_000)    
        number_price2 = st.number_input("До:", min_value=vals[0],
                                 max_value=vals[1], step=100_000, value=vals[1])  
        opts.update({'price_value': [number_price1, number_price2]})
    with cols_house_fl1[1]:
        vals = new_di_vals_float['area_value']
        number_sq1 = st.number_input(":green[Площадь квартиры], от:", min_value=np.float64(vals[0]),
                                 max_value=vals[1], step=np.float64(0.1))    
        number_sq2 = st.number_input("До:", min_value=np.float64(vals[0]),
                                 max_value=vals[1], step=np.float64(0.1), value=vals[1])  
        opts.update({'area_value': [number_sq1, number_sq2]})
    metro_dist = {"Все": [-2, 1000],
                  "До 5 минут": [1, 5],
                  "До 10 минут": [1, 10],
                  "До 15 минут": [1, 15],
                  "До 20 минут": [1, 20],
                  "Более 20 минут": [21, 1000],
                  "нет информации": [-1, -1]}
    with cols_house_fl1[2]:
        distMetro = st.selectbox(
                    f"Выберите :green[Время в пути до метро, пешком]:",
                    options=metro_dist,
                    index=None,
                    placeholder="Выберите опцию")
        if distMetro is not None:
            if len(distMetro) > 0:
                opts.update({'location_metro_time-on-foot': metro_dist[distMetro]})
                
    # Время в пути до метро, на машине

    
    ################################# Параметры дома #################################
    st.subheader(":rainbow[Параметры дома]")
    cat_cols_build = ['built-year', 'ready-quarter']
    cols_house = st.columns(len(cat_cols_build), gap='large')
    cnt = 0
    for key in cat_cols_build:
        ln = len(new_di_vals[key])
        chosen_vals = [val for val in st.session_state[key] if val in new_di_vals[key]]
        with cols_house[cnt]:
            options = st.multiselect(
                f"Выберите :green[{cols[key]}] :gray[(осталось вариантов {ln})]:",
                new_di_vals[key],
                chosen_vals,
                placeholder="Выберите опцию")
        opts.update({key: set(options)})
        cnt = cnt + 1
        
    ################################# Геолокация #################################
    st.subheader(":rainbow[Геолокация]")
    # dropped 'location_sub-locality-name'
    cat_cols_geo = ['location_region', 'location_locality-name', 'location_non-admin-sub-locality', 'location_address', 'location_metro_name', 'building-name']
    cols_geo = st.columns([0.3, 0.7], gap='small') # len(cat_cols_geo)
    cnt = 0
    for key in cat_cols_geo:
        ln = len(new_di_vals[key])
        chosen_vals = [val for val in st.session_state[key] if val in new_di_vals[key]]
        with cols_geo[cnt]:
            options = st.multiselect(
                f"Выберите :green[{cols[key]}] :gray[(осталось вариантов {ln})]:",
                new_di_vals[key],
                chosen_vals,
                placeholder="Выберите опцию")
        opts.update({key: set(options)})
    cnt = cnt + 1
    
    # отрисовка карты
    # scatter_map scatter_mapbox   
    geo_data["color"] = pd.qcut(geo_data["to_count"].values, q=5, duplicates="drop") # , labels=range(1, 5 + 1)
    if len(set(geo_data["color"])) > 1:
        geo_data["color"] = [(i.right + i.left)/2 for i in geo_data["color"].values]
    else:
        geo_data["color"] = None
    geo_data.rename(columns={"to_count": "кол-во предложений"}, inplace=True)  
    print(geo_data.head())
   
    with cols_geo[cnt]:
        fig = px.scatter_mapbox(
            geo_data,
            lat="location_latitude",
            lon="location_longitude",
            color="color",
            color_continuous_scale = "jet", # viridis rdgy electric emrld jet hsv
            size="кол-во предложений",
            size_max=15,
            zoom=10,
            mapbox_style="open-street-map", # carto-darkmatter carto-positron open-street-map
            text=geo_data["building-name"].astype(str),  # this is the text for labels - building-name location_address
            height=700,
            labels={'labels': 'building-name'} # , 'color': 'rgb(100, 100 ,100)' building-name location_address
        )
        fig.update_layout(
            font_family="Courier New",
            font_color="rgb(50,50,50)",
        )
        geo_vals = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points", "box", "lasso"])
        geo_vals_list = [i["text"] for i in geo_vals["selection"]["points"]]
        if len(geo_vals_list) > 0:
            opts.update({"building-name": geo_vals_list}) # building-name location_address

    ################################# Прочее #################################
    st.subheader(":rainbow[Прочие параметры квартиры и дома]")
    cat_cols_other = ['renovation', 'balcony', 'bathroom-unit']
    cols_house = st.columns(len(cat_cols_other), gap='large')
    cnt = 0
    for key in cat_cols_other:
        ln = len(new_di_vals[key])
        chosen_vals = [val for val in st.session_state[key] if val in new_di_vals[key]]
        with cols_house[cnt]:
            options = st.multiselect(
                f"Выберите :green[{cols[key]}] :gray[(осталось вариантов {ln})]:",
                new_di_vals[key],
                chosen_vals,
                placeholder="Выберите опцию")
        opts.update({key: set(options)})
        cnt = cnt + 1  
    cat_cols_build_other = ['floors-total', 'lift', 'parking', 'building-type']
    cols_house = st.columns(len(cat_cols_build_other), gap='large')
    cnt = 0
    for key in cat_cols_build_other:
        ln = len(new_di_vals[key])
        chosen_vals = [val for val in st.session_state[key] if val in new_di_vals[key]]
        with cols_house[cnt]:
            options = st.multiselect(
                f"Выберите :green[{cols[key]}] :gray[(осталось вариантов {ln})]:",
                new_di_vals[key],
                chosen_vals,
                placeholder="Выберите опцию")
        opts.update({key: set(options)})
        cnt = cnt + 1
    
    # отрисовка набора данных
    st.subheader(f":rainbow[Предложения] - {dataset.shape[0]}")    
    if "Unnamed: 0" in dataset.columns.to_list():
        dataset.drop(columns=["Unnamed: 0"], inplace=True)
    to_show = dataset[['building-name',  'renovation', 'description',
       'rooms', 'balcony', 'bathroom-unit', 'floor',  'floors-total', 'built-year', 'ready-quarter', 'lift',
       'parking', 'location_address', 'price_value', 'area_value', 'living-space_value', 'location_metro_name',
       'location_metro_time-on-foot',
       'location_metro_time-on-transport']]
    st.dataframe(to_show.head(100).style.highlight_max(axis=0), hide_index=True)
    
    return opts
        
def main():
    # print("1st state: ", st.session_state)
    dataset, categorical_vals, float_vals = load_full_dataset()
    st.button("Сброс фильтров", on_click=reset_state_callback)
    current_query = get_ui(cols, dataset, categorical_vals, float_vals)
    update_state(current_query, categorical_vals)
    print("st.session_state.counter: ", st.session_state.counter)

st.set_page_config(layout="wide")
initialize_state() 
main()
    