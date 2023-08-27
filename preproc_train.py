import pandas as pd


def merge_data(area_file, areatype_file, building_file, district_file, geonim_file, subrf_file, town_file, prefix_file):

    def preprocessing(file_name, suffix):
        data = pd.read_csv(file_name)
        data = data.add_suffix(suffix)
        return data

    area = preprocessing(area_file, '_area')
    areatype = preprocessing(areatype_file, '_areatype')
    building = preprocessing(building_file, '_building')
    district = preprocessing(district_file, '_district')
    geonim = preprocessing(geonim_file, '_geonim')
    subrf = preprocessing(subrf_file, '_subrf')
    town = preprocessing(town_file, '_town')
    prefix = preprocessing(prefix_file, '_prefix')
    building = building[building['is_actual_building'] == True]

    merged_area = pd.merge(area, areatype, left_on='type_id_area', right_on='id_areatype', how='left')
    merged_area.drop(['type_id_area', 'id_areatype'], axis=1, inplace=True)
    merged_prefix = pd.merge(prefix, merged_area, left_on='area_id_prefix', right_on='id_area', how='left')
    merged_prefix = pd.merge(merged_prefix, geonim, left_on='geonim_id_prefix', right_on='id_geonim', how='left')
    merged_prefix = pd.merge(merged_prefix, subrf, left_on='sub_rf_id_prefix', right_on='id_subrf', how='left')
    merged_prefix = pd.merge(merged_prefix, town, left_on='town_id_prefix', right_on='id_town', how='left')
    merged_prefix.drop(['area_id_prefix', 'id_area', 'geonim_id_prefix', 'id_geonim', 'sub_rf_id_prefix', 'id_subrf', 'town_id_prefix', 'id_town'], axis=1, inplace=True)
    merged_data = pd.merge(building, merged_prefix, left_on='prefix_id_building', right_on='id_prefix')
    merged_data = pd.merge(merged_data, district, left_on='district_id_building', right_on='id_district')
    merged_data.drop(['prefix_id_building', 'id_prefix', 'district_id_building', 'id_district'], axis=1, inplace=True)
    return merged_data
