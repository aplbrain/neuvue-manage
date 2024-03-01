import json
import numpy as np
from nglui.parser import extract_multicut
import collections
import requests
import backoff
import pandas as pd
from nglui.statebuilder import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    AnnotationLayerConfig,
    LineMapper,
    PointMapper,
    StateBuilder,
    ChainedStateBuilder,
    BoundingBoxMapper
    )

from utils.constants import NEUVUE_NG_URL, IMAGE, SEG, MINNIE_RESOLUTION
from itertools import repeat
from utils.chunkedgraph_utils import *
import random


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def extract_multicut_points(state, seg_layer='seg'):
    """
    :param seg_layer:
    :param state: JSON ordered dict
    :return: source points, sink points, segmentaiton ID, neuroglancer state
    """
    if isinstance(state, str):
        state = json.loads(state, object_pairs_hook=collections.OrderedDict)

    pts, side, svid, root_id = extract_multicut(state, seg_layer=seg_layer)

    source_pts = pts[np.where(side == 'source')]
    sink_pts = pts[np.where(side == 'sink')]

    return source_pts, sink_pts, root_id, state


def get_coordinate(ng_state):
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)

    coords = ng_state['navigation']['pose']['position']['voxelCoordinates']

    return coords


def get_annotations(ng_state, layer_name):
    '''

    :param ng_state: neuroglancer state
    :param layer_name: str, name of layer to extract annotations from
    :return: dataframe of annotations
    '''
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)

    annos = []
    for layer in ng_state['layers']:
        if layer['name'] == layer_name:
            for point in layer['annotations']:
                annos.append(point['point'])

    return pd.DataFrame(data={'annos': annos})


def get_annotations_and_descriptions(ng_state, layer_name):
    '''

    :param ng_state: neuroglancer state
    :param layer_name: str, name of layer to extract annotations from
    :return: dataframe of annotations
    '''
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)

    annos = []
    descriptions = []
    for layer in ng_state['layers']:
        if layer['name'] == layer_name:
            for point in layer['annotations']:
                annos.append(point['point'])
                descriptions.append(point['description'])

    return pd.DataFrame(data={'annos': annos, 'descriptions': descriptions})


def get_annotations_and_tags(ng_state, layer_name):
    '''

    :param ng_state: neuroglancer state
    :param layer_name: str, name of layer to extract annotations from
    :return: dataframe of annotations
    '''
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)
    tag_labels = {}
    annos = []
    tags = []
    for layer in ng_state['layers']:
        if layer['name'] == layer_name:
            for point in layer['annotations']:
                annos.append(point['point'])
                if len(point['tagIds']) > 0:
                    tags.append(point['tagIds'][0])
                else:
                    tags.append(0)

            for label in layer['annotationTags']:
                tag_labels[label['label']] = label['id']

    return pd.DataFrame(data={'annos': annos, 'tags': tags}), tag_labels


def get_anno_pairs(ng_state, layer_name):
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)
    anno_pairs = []
    sv_pairs = []
    for layer in ng_state['layers']:
        if layer['name'] == layer_name:
            for point in layer['annotations']:
                anno_pairs.append([point['pointA'], point['pointB']])
                sv_pairs.append(point['segments'])

    return anno_pairs, sv_pairs


def get_segmentation_ids(ng_state, layer_name='seg', include_hidden=True):
    '''

    :param ng_state: neuroglancer state
    :param layer_name: str, name of layer to extract segmentation ids from
    :return: list of segmentations
    '''
    ng_state = json.loads(ng_state, object_pairs_hook=collections.OrderedDict)

    segments = []
    hidden_segs = []
    for layer in ng_state['layers']:
        if layer['name'] == layer_name:
            segments = layer['segments']
            try:
                hidden_segs = layer['hiddenSegments']
            except:
                pass
        else:
            pass

    if include_hidden:
        for x in hidden_segs:
            if x not in segments:
                segments.append(x)

    return segments


def build_basic_ng_state(seg_id, coordinate, img_src=IMAGE, seg_src=SEG, em_zoom=26):
    '''

    :param seg_id: segmentation id
    :param coordinate: coordinate to center the view on
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=[str(seg_id)],
        active=True,
        view_kws=segmentation_view_options
    )

    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    state = StateBuilder(layers=[img_layer, seg_layer], view_kws=view_options).render_state(return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['layers'][1].update({'tab': 'graph'})

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state


def build_basic_ng_state_with_seg_colors(seg_id_list, seg_color_list, coordinate, img_src=IMAGE, seg_src=SEG, em_zoom=26):
    '''

    :param seg_id: segmentation id
    :param coordinate: coordinate to center the view on
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=seg_id_list,
        fixed_id_colors=seg_color_list,
        active=True,
        view_kws=segmentation_view_options
    )

    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    state = StateBuilder(layers=[img_layer, seg_layer], view_kws=view_options).render_state(return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['layers'][1].update({'tab': 'graph'})

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state


def add_basic_ng_states_to_df(df, root_id_str='pt_root_id', pt_position_str='pt_position'):
    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    df['states'] = list(map(build_basic_ng_state, seg_ids, coords))


def add_basic_ng_states_with_anno_layers_to_df(df, root_id_str='pt_root_id', pt_position_str='pt_position', anno_names=['annotations'], anno_colors=['#FB00FF']):
    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    df['states'] = list(map(build_basic_ng_state_with_anno_layers, seg_ids, coords, repeat(anno_names), repeat(anno_colors)))


def add_basic_ng_states_with_anno_layer_and_points_from_ground_truth_to_df(df, root_id_str='pt_root_id', pt_position_str='pt_position', state_str='ng_states', anno_point_layer_name='annos', anno_name='annotations', tags=[]):
    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    states = list(df[state_str].values)
    annos = [get_annotations(state, anno_point_layer_name) for state in states]
    df['states'] = list(map(build_basic_ng_state_with_anno_layer_and_points, seg_ids, coords, annos, repeat(anno_name), repeat(tags)))

def add_basic_ng_states_with_anno_layer_and_points_to_df(df, root_id_str='pt_root_id', pt_position_str='pt_position', anno_name='annotations', annos_str='annos', tags=[]):
    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    annos = [pd.DataFrame(data={'annos': x}) for x in df[annos_str].values]
    df['states'] = list(map(build_basic_ng_state_with_anno_layer_and_points, seg_ids, coords, annos, repeat(anno_name), repeat(tags)))

def add_basic_agents_states(
    df, 
    root_id_str='pt_root_id', 
    hidden_seg_id_str = 'hidden_ids', 
    pt_position_str='pt_position', 
    anno_name='annotations',
    annos_str='annos',
    bounding_box_name=None,
    bb_str = 'bb',
    tags=[]):

    seg_ids = list(df[root_id_str].values)
    hidden_seg_ids = []
    if hidden_seg_id_str in df.columns:
        hidden_seg_ids = list(df[hidden_seg_id_str].values)
    coords = list(df[pt_position_str].values)
    anno_dfs = [pd.DataFrame(data={'annos': x}) for x in df[annos_str].values]
    bounding_box_dfs = [pd.DataFrame(data={'bb0': [x[0]], 'bb1': [x[1]]}) for x in df[bb_str].values]
    df['states'] = list(map(
        build_agents_state, 
        seg_ids, 
        coords, 
        anno_dfs, 
        bounding_box_dfs,
        hidden_seg_ids, 
        repeat(anno_name), 
        repeat(bounding_box_name), 
        repeat(tags)
        ))

def add_basic_endpoint_states(
    df, 
    root_id_str='pt_root_id', 
    pt_position_str='pt_position', 
    anno_name='annotations',
    annos_str='annos',
    bounding_box_name=None,
    bb_str = 'bb',
    tags=[]):

    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    anno_df = [pd.DataFrame(data={'annos': x}) for x in df[annos_str].values]
    bounding_box_df = [pd.DataFrame([{'bb0': y[0], 'bb1': y[1]} for y in x]) for x in df[bb_str].values]
    df['states'] = list(map(
        build_endpoint_state, 
        seg_ids, 
        coords, 
        anno_df, 
        bounding_box_df,
        repeat(anno_name), 
        repeat(bounding_box_name), 
        repeat(tags)
        ))


def add_basic_ng_states_with_anno_layer_points_description_from_ground_truth_to_df(df, root_id_str='pt_root_id', pt_position_str='pt_position', state_str='ng_states', anno_point_layer_name='annos', anno_name='annotations'):
    seg_ids = list(df[root_id_str].values)
    coords = list(df[pt_position_str].values)
    states = list(df[state_str].values)
    annos = [get_annotations_and_descriptions(state, anno_point_layer_name) for state in states]
    df['states'] = list(map(build_basic_ng_state_with_anno_layer_and_points_description, seg_ids, coords, annos, repeat(anno_name)))


def build_basic_ng_state_with_anno_layers(seg_id, coordinate, anno_names=['annotations'], anno_colors=['#FB00FF'],img_src=IMAGE, seg_src=SEG, em_zoom=26):
    '''

    :param seg_id: segmentation id
    :param coordinate: coordinate to center the view on
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=[str(seg_id)],
        active=True,
        view_kws=segmentation_view_options
    )

    layers = [img_layer, seg_layer]
    for i in range(len(anno_names)):
        anno_layer = AnnotationLayerConfig(name=anno_names[i], color=anno_colors[i], linked_segmentation_layer='seg')
        layers.append(anno_layer)
    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    state = StateBuilder(layers=layers, view_kws=view_options).render_state(return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['selectedLayer']['layer'] = anno_names[0]

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state


def build_basic_ng_state_with_anno_layer_and_points(seg_id, coordinate, anno_point_df, anno_name='annotations', tags=[], img_src=IMAGE, seg_src=SEG, em_zoom=26):
    '''
    :param seg_id: segmentation id
    :param coordinate: coordinate to center the view on
    :param anno_point_df: df with annotation points in column named 'annos'
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    if type(seg_id) is not list:
        seg_id = [str(seg_id)]

    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=seg_id,
        active=True,
        view_kws=segmentation_view_options
    )

    points = PointMapper(point_column='annos')
    anno_layer = AnnotationLayerConfig(name=anno_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=points,  tags=tags)

    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    state = StateBuilder(layers=[img_layer, seg_layer, anno_layer], view_kws=view_options).render_state(anno_point_df, return_as='json', url_prefix=NEUVUE_NG_URL)
    state_dict = json.loads(state)

    state_dict['selectedLayer']['layer'] = 'seg'
    state_dict['layers'][1].update({'tab': 'graph'})

    if 'jsonStateServer' not in state_dict.keys():
        state_dict["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state_dict)
    return state

def build_agents_state(
    seg_id, 
    coordinate, 
    anno_point_df, 
    bounding_box_df,
    hidden_segments=[], 
    anno_name='annotations',
    bounding_box_name='bounding_box',
    tags=[], 
    img_src=IMAGE, 
    seg_src=SEG, 
    em_zoom=26,
    ):
    '''

    :param seg_id: segmentation id
    :param hidden_segments: list of hidden segment ids
    :param coordinate: coordinate to center the view on
    :param anno_point_df: df with annotation points in column named 'annos'
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    if type(seg_id) is not list:
        seg_id = [str(seg_id)]

    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=seg_id,
        active=True,
        view_kws=segmentation_view_options
    )

    points = PointMapper(point_column='annos')
    anno_layer = AnnotationLayerConfig(name=anno_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=points,  tags=tags)
    false_merge_layer = AnnotationLayerConfig(name='false_merges', color='#DC758F', linked_segmentation_layer='seg')
    base_layers = [img_layer, seg_layer, anno_layer, false_merge_layer]
    
    bounding_box = BoundingBoxMapper(
        point_column_a='bb0',
        point_column_b='bb1'
    )
    bounding_box_layer = AnnotationLayerConfig(name=bounding_box_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=bounding_box)
    bounding_box_state = StateBuilder(layers=[bounding_box_layer], resolution=MINNIE_RESOLUTION)
    
    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    base_state = StateBuilder(layers=base_layers, view_kws=view_options)

    chained_state = ChainedStateBuilder([base_state, bounding_box_state])
    state = chained_state.render_state([anno_point_df, bounding_box_df], return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['selectedLayer']['layer'] = 'seg'
    
    hidden_segments_list = hidden_segments
    if type(hidden_segments_list) != list:
        hidden_segments_list = hidden_segments_list.tolist()
    state['layers'][1]['hiddenSegments'] = hidden_segments_list
    #state['layers'][1].update({'tab': 'render'})
    
    # create color dictionary
    meshColors = {seg_id[0] : '#4290BD'}
    candidate_colors = ['#DBC906', '#FDD000', '#F6A702', '#FC7A21', '#F53F15']
    candidate_list = [seg_id[-1]]
    for seg in hidden_segments_list:
        candidate_list.append(seg)

    for i in range(0, len(candidate_list)):
        if i >= len(candidate_colors):
            r = lambda: random.randint(0,255)
            meshColors[candidate_list[i]] = ('#%02X%02X%02X' % (r(),r(),r()))
        else:
            meshColors[candidate_list[i]] = candidate_colors[i]
    state['layers'][1]['segmentColors'] = meshColors
    

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state

def build_endpoint_state(
    seg_id, 
    coordinate, 
    anno_point_df, 
    bounding_box_df,
    anno_name='annotations',
    bounding_box_name='bounding_box',
    tags=[], 
    img_src=IMAGE, 
    seg_src=SEG, 
    em_zoom=26,
    ):
    '''

    :param seg_id: segmentation id
    :param hidden_segments: list of hidden segment ids
    :param coordinate: coordinate to center the view on
    :param anno_point_df: df with annotation points in column named 'annos'
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    if type(seg_id) is not list:
        seg_id = [str(seg_id)]

    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=seg_id,
        active=True,
        view_kws=segmentation_view_options
    )

    points = PointMapper(point_column='annos')
    anno_layer = AnnotationLayerConfig(name=anno_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=points,  tags=tags)
    false_merge_layer = AnnotationLayerConfig(name='false_merges', color='#DC758F', linked_segmentation_layer='seg')
    base_layers = [img_layer, seg_layer, anno_layer, false_merge_layer]
    
    bounding_box = BoundingBoxMapper(
        point_column_a='bb0',
        point_column_b='bb1'
    )
    bounding_box_layer = AnnotationLayerConfig(name=bounding_box_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=bounding_box)
    bounding_box_state = StateBuilder(layers=[bounding_box_layer], resolution=MINNIE_RESOLUTION)
    
    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    base_state = StateBuilder(layers=base_layers, view_kws=view_options)

    chained_state = ChainedStateBuilder([base_state, bounding_box_state])
    state = chained_state.render_state([anno_point_df, bounding_box_df], return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['selectedLayer']['layer'] = 'seg'
    
    #state['layers'][1].update({'tab': 'render'})
    
    # create color dictionary
    meshColors = {seg_id[0] : '#4290BD'}
    state['layers'][1]['segmentColors'] = meshColors
    

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state


def build_basic_ng_state_with_anno_layer_and_points_description(seg_id, coordinate, anno_point_df, anno_name='annotations', img_src=IMAGE, seg_src=SEG, em_zoom=26):
    '''

    :param seg_id: segmentation id
    :param coordinate: coordinate to center the view on
    :param anno_point_df: df with annotation points in column named 'annos'
    :param img_src: img url
    :param seg_src: segmentation url
    :param em_zoom: zoom level for image data
    :return: neuroglancer state
    '''
    # Create ImageLayerConfig
    black = .35
    white = .7

    img_layer = ImageLayerConfig(
        name='em',
        source=img_src,
        contrast_controls=True,
        black=black,
        white=white
    )

    # Create SegmentationLayerConfig
    segmentation_view_options = {
        'alpha_selected': 0.6,
        'alpha_3d': 0.6
    }
    seg_layer = SegmentationLayerConfig(
        name='seg',
        source=seg_src,
        fixed_ids=seg_id,
        active=True,
        view_kws=segmentation_view_options
    )

    points = PointMapper(point_column='annos', description_column='descriptions')
    anno_layer = AnnotationLayerConfig(name=anno_name, color='#FB00FF', linked_segmentation_layer='seg', mapping_rules=points)

    view_options = {'position': coordinate, 'zoom_image': em_zoom}
    state = StateBuilder(layers=[img_layer, seg_layer, anno_layer], view_kws=view_options).render_state(anno_point_df, return_as='dict', url_prefix=NEUVUE_NG_URL)
    state['selectedLayer']['layer'] = anno_name

    if 'jsonStateServer' not in state.keys():
        state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

    state = json.dumps(state)
    return state


def add_split_preview_states_to_df(df, cave_client):
    """
    maps the ng_state to the row in the df and adds the split preview to the state
    """
    HEADER = cave_client.auth.request_header
    states = df['ng_state'].tolist()
    df['states'] = list(map(get_split_preview, states, repeat(cave_client), repeat(26.0)))

    return df


def get_split_preview(state, cave_client, zoom_factor=26.0, minimal_preview=True):
    # get the multi cut points
    try:
        source_pts, sink_pts, root_id, state = extract_multicut_points(state)
    except requests.exceptions.HTTPError as err:
        return err.response.text
    except requests.exceptions.RetryError as err:
        return 'Retry Error'

    # convert to voxels
    x_pts = np.append(source_pts[:, 0], sink_pts[:, 0])
    y_pts = np.append(source_pts[:, 1], sink_pts[:, 1])
    z_pts = np.append(source_pts[:, 2], sink_pts[:, 2])

    # set bounds around the multi cut points
    bounds = np.array([[(np.min(x_pts))/8, (np.max(x_pts))/8],
                       [(np.min(y_pts))/8, (np.max(y_pts))/8],
                       [(np.min(z_pts))/40, (np.max(z_pts))/40]], dtype=int)

    # get the supervoxels for the seg id and bounds
    try:
        local_sv = get_leaves(cave_client=cave_client, root_id=root_id, bounds=bounds)
    except requests.exceptions.HTTPError as err:
        return err.response.text
    except requests.exceptions.RetryError as err:
        return 'Retry Error'

    try:
        # preview the split and get the supervoxels for each side
        source_cc, sink_cc, success = preview_split(cave_client=cave_client, source_points=source_pts,
                                                    sink_points=sink_pts, root_id=root_id)
    except requests.exceptions.HTTPError as err:
        return err.response.text
    except requests.exceptions.RetryError as err:
        return 'Retry Error'

    # split preview is successful
    if success:

        if minimal_preview:
            # filter to just the local sv in the bounds
            local_source_cc = [x for x in source_cc if x in local_sv]
            local_sink_cc = [x for x in sink_cc if x in local_sv]

            # get layer 2 ids
            try:
                source_layer_2 = get_layer2_ids(cave_client=cave_client, local_cc=local_source_cc)
            except requests.exceptions.HTTPError as err:
                return err.response.text
            except requests.exceptions.RetryError as err:
                return 'Retry Error'

            try:
                sink_layer_2 = get_layer2_ids(cave_client=cave_client, local_cc=local_sink_cc)
            except requests.exceptions.HTTPError as err:
                return err.response.text
            except requests.exceptions.RetryError as err:
                return 'Retry Error'

            # get which layer 2 ids are on both side
            overlap = [x for x in source_layer_2 if x in sink_layer_2]
            overlap_unique = np.unique(overlap)

            # list of layer 2 red/blue only ids
            # source_layer_2_view = [str(x) for x in np.unique(source_layer_2) if x not in overlap_unique]
            # sink_layer_2_view = [str(x) for x in np.unique(sink_layer_2) if x not in overlap_unique]

            # make filter for sv ids in overlap
            indexes = [index for index in range(len(source_layer_2)) if source_layer_2[index] in overlap_unique]
            source_sv = list(np.array(local_source_cc)[indexes])
            source_sv = [str(x) for x in source_sv]

            indexes = [index for index in range(len(sink_layer_2)) if sink_layer_2[index] in overlap_unique]
            sink_sv = list(np.array(local_sink_cc)[indexes])
            sink_sv = [str(x) for x in sink_sv]

            # # colors for the layer 2 ids
            # seg_colors = {}
            # for svid in source_layer_2_view:
            #     # red
            #     seg_colors[svid] = '#ff0000'
            #
            # for svid in sink_layer_2_view:
            #     # blue
            #     seg_colors[svid] = '#0008ff'
            #
            # state['layers'][1].update({'segmentColors': seg_colors})
            #
            # segments = state['layers'][1]['segments']
            # segments = segments + source_layer_2_view + sink_layer_2_view
            # state['layers'][1]['segments'] = segments
        else:
            source_sv = source_cc
            sink_sv = sink_cc

        # supervoxel colors
        seg_colors_ws = {}
        for svid in source_sv:
            # red
            seg_colors_ws[svid] = '#ff0000'

        for svid in sink_sv:
            # blue
            seg_colors_ws[svid] = '#0008ff'

        # add the watershed layer
        ws_layer = {
            "source": "precomputed://s3://bossdb-open-data/iarpa_microns/minnie/minnie65/ws",
            "type": "segmentation",
            "selectedAlpha": 0.5,
            "segmentColors": seg_colors_ws,
            "segments": source_sv + sink_sv,
            "skeletonRendering": {
                "mode2d": "lines_and_points",
                "mode3d": "lines"
            },
            "name": "split_preview"
        }
        state['layers'].append(ws_layer)

        # reset the segmentation view to be on the Rendering tab
        state['layers'][1].pop('tab')
        state['selectedLayer']['layer'] = 'split_preview'

        # set the zoom level
        state['navigation']['zoomFactor'] = zoom_factor

        remove_layer = None
        for i in range(len(state['layers'])):
            if state['layers'][i]['name'] == 'split_location':
                state['layers'][i]['annotationColor'] = "#f99008"
            elif state['layers'][i]['name'] == 'not_cut_path':
                remove_layer = i

        if remove_layer is not None:
            state['layers'].pop(remove_layer)

        if 'jsonStateServer' not in state.keys():
            state.update({"jsonStateServer": "https://global.daf-apis.com/nglstate/api/v1/post"})

        state = json.dumps(state)

        return state
    else:
        return 'Split Failed'
