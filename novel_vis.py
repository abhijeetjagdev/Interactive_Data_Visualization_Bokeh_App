import pandas as pd
import numpy as np

from collections import Counter

from bokeh.layouts import row, column, widgetbox, Row, Column, gridplot, layout
from bokeh.models import Select
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure, Figure
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, Legend
from bokeh.models.glyphs import Circle
from bokeh.models.widgets import CheckboxButtonGroup, RadioButtonGroup, Paragraph
from bokeh.charts import BoxPlot
from bokeh.models import Range1d, FuncTickFormatter
from bokeh.models.sources import ColumnDataSource
# from bokeh.sampledata.autompg import autompg

# df = autompg.copy()

# xcols = ['nihss_BL', 'aspects_sitebl', 'onsettoct_time', 'onsettopuncture_time', 'picturetopuncture_time', 'puncturetoperfusion_time', 'mtici_cent', 'comorbidity_count']
xnames = {"Baseline NIHSS": 'nihss_BL', 'Baseline MRS': 'hx_mrs', 'Onset to CT time': 'onsettoct_time', 'Onset to Randomization time': 'onsettorand_time',
          'Door to CT time': 'doortoct_time',
          'Onset to puncture time': 'onsettopuncture_time',
          "CT to puncture time": 'picturetopuncture_time', "CT to reperfusion time": 'picturetoperfusion_time', "Door to puncture time": 'doortopuncture_time',
          'Puncture to reperfusion time': 'puncturetoperfusion_time', 'Onset to Perfusion time': 'onsettoperfusion_time',
          'mTICI Reperfusion score': 'mtici_cent', "Count of comorbidities": 'comorbidity_count',
          "mTICI Reperfusion score": "mtici_cent", 'Age': 'age_calc', 'ASPECTS Score (Baseline)': "aspects_sitebl"}

# ycols = ['nihss_D90', 'mrs_D90', 'mtici_cent']
ynames = {"NIHSS-Day 90": 'nihss_D90', "NIHSS-Day 30": "nihss_D30", "NIHSS-Day 5": "nihss_D5", "Baseline NIHSS": "nihss_BL",
          "NIHSS-48 Hours": "nihss_H48", "NIHSS-24 Hours": "nihss_H24", "NIHSS- 2-8 Hours": "nihss_H2_8",
          "MRS-Day 90": 'mrs_D90', "MRS-Day 30": 'mrs_D30', "mTICI Reperfusion score": 'mtici_cent'}

# pre_conditions = ['anticoagulation_any', 'ivtpa_given', 'hxcad', 'hxchf', 'hxstrk_r',
#                   'hxstrk_p', 'hxich', 'hxheadspinetrauma',
#                   'hxmajsurg', 'hxpvd', 'hxhighchol', 'hxafib',
#                   'hxhtn', 'hxdiab', 'hxsmk']

# condition_names = {'anticoagulation_any': "Blood Thinners", 'ivtpa_given': "TPA Given at hospital", 'hxcad': "Coronary Artery Disease", 'hxchf': "Congestive Heart Failure",
#                    'hxstrk_r': "Previous Stroke_r",
#                    'hxstrk_p': "Previous Stroke_p", 'hxich': "Previous Intracereberal Hemmorage", 'hxheadspinetrauma': "Spine/Head Trauma",
#                    'hxmajsurg': "Previous major surgery", 'hxpvd': "Peripheral Vascular Disease", 'hxhighchol': "High Cholesterol", 'hxafib': "Atrial Fibrilation",
#                    'hxhtn': "Hypertension", 'hxdiab': "Diabetes", 'hxsmk': "Smoking"}

pre_conditions = ['ivtpa_given', 'hxafib_anticoag', 'hxhighchol', 'hxafib',
                  'hxhtn', 'hxdiab', 'hxsmk']

condition_names = {'ivtpa_given': "tPA Given at hospital", 'hxafib_anticoag': "Blood Thinners (Afib)",
                   'hxhighchol': "High Cholesterol", 'hxafib': "Atrial Fibrilation",
                   'hxhtn': "Hypertension", 'hxdiab': "Diabetes", 'hxsmk': "Smoking"}


def comorbidity_count(r):
    count = 0
    for cond in pre_conditions:
        if cond is not 'ivtpa_given' and r[cond] is 1:
            count += 1
    return count


raw_data = pd.read_excel("ESCAPE_visulalization dataset12.xlsx")
# create anticoagulation (any kind) column
# raw_data['anticoagulation_any'] = raw_data.apply(lambda r: r['anticoagulation_none'] == 0, axis=1)
raw_data['comorbidity_count'] = raw_data.apply(comorbidity_count, axis=1)
# df = raw_data[['treatment', 'onsettoct_time', 'onsettopuncture_time', 'onsettotpa_time', 'picturetopuncture_time', 'puncturetoperfusion_time',
#               'nihss_BL', 'nihss_D90', 'aspects_sitebl', 'mrs_D90', 'mtici_cent', 'comorbidity_count'] + pre_conditions]
# list(set(list(xnames.keys())+list(ynames.keys())+pre_conditions))

# Calculation patient outcome MRS relative to patients in the same bin of initial NIHSS
groups = raw_data.groupby(pd.cut(raw_data['nihss_BL'], 5))

# Include all needed columns, removing duplicates by casting to set. TODO: Include size columns?
df = raw_data[list(set(list(xnames.values()) + list(ynames.values()) + pre_conditions)) + ['treatment', 'sex']]

SIZES = list(range(6, 22, 3))
COLORS = Spectral5
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"
color_t = "red"
color_c = "blue"

columns = sorted(df.columns)
discrete = [x for x in columns if df[x].dtype == object]
continuous = [x for x in columns if x not in discrete]
quantileable = [x for x in continuous if len(df[x].unique()) > 20]


def filter_no_prexisting_condition(r):
    for i in pre_conditions:
        if r[i] == 1:
            return False
        else:
            return True


def create_data_sources():
    # only use data as selected
    # treatment
    if (treatment.active == 0):
        # treatment group only
        df_filtered = df[df['treatment'] == 1]
    elif (treatment.active == 1):
        # control group only
        df_filtered = df[df['treatment'] == 0]
    else:
        df_filtered = df

    # filter by sex TODO: check these are the right numbers
    if (sex.active == 0):
        # male
        df_filtered = df_filtered[df_filtered['sex'] == 1]
    elif (sex.active == 1):
        # female
        df_filtered = df_filtered[df_filtered['sex'] == 2]
    else:
        pass

    for i in range(0, len(pre_conditions)):
        if conditions_controls[i].active == 1:  # yes selected
            df_filtered = df_filtered[df_filtered[pre_conditions[i]] == 1]
        elif conditions_controls[i].active == 2:  # no selected
            df_filtered = df_filtered[df_filtered[pre_conditions[i]] == 0]
        # 0 being selected means 'Either', so don't filter at all

    df_filtered_control = df_filtered[df_filtered['treatment'] == 0]
    df_filtered_treatment = df_filtered[df_filtered['treatment'] == 1]

    xs = df_filtered[xnames[x.value]].values
    # print("{0}: {1}".format(xnames[x.value], xs))
    xs_treatment = df_filtered_treatment[xnames[x.value]].values
    xs_control = df_filtered_control[xnames[x.value]].values

    ys = df_filtered[ynames[y.value]].values
    ys_treatment = df_filtered_treatment[ynames[y.value]].values
    ys_control = df_filtered_control[ynames[y.value]].values

    # filter data to remove NaNs
    xs_treatment_nonan = xs_treatment[~np.isnan(xs_treatment)]
    xs_control_nonan = xs_control[~np.isnan(xs_control)]
    ys_treatment_nonan = ys_treatment[~np.isnan(ys_treatment)]
    ys_control_nonan = ys_control[~np.isnan(ys_control)]

    # Zip the points into tuples and then get a count of the unique points
    scatter_points_treatment = zip(xs_treatment_nonan, ys_treatment_nonan)
    scatter_points_control = zip(xs_control_nonan, ys_control_nonan)
    counter_points_treatment = Counter(scatter_points_treatment)
    counter_points_control = Counter(scatter_points_control)

    # Add count into the tuple
    treatment_tuples = [a + (counter_points_treatment[a],) for a in counter_points_treatment]
    control_tuples = [a + (counter_points_control[a],) for a in counter_points_control]

    # Project the tuples into lists
    projection_treatment = list(zip(*treatment_tuples))
    projection_control = list(zip(*control_tuples))

    size_factor = 9

    if projection_treatment:  # if the list is not empty: ie: there are treatment points to map
        treatment_x = [float(i) for i in list(projection_treatment[0])]
        treatment_y = [float(i) for i in list(projection_treatment[1])]
        treatment_count = list(projection_treatment[2])
        treatment_size = [size_factor * item for item in list(projection_treatment[2])]
        treatment_source = ColumnDataSource(
            data=dict(
                x=treatment_x,
                y=treatment_y,
                size=treatment_size,
                count=treatment_count
            )
        )
    else:
        treatment_source = None

    if projection_control:  # if the list is not empty: ie: there are control points to map
        control_x = [float(i) for i in list(projection_control[0])]
        control_y = [float(i) for i in list(projection_control[1])]
        control_count = list(projection_control[2])
        control_size = [size_factor * item for item in list(projection_control[2])]
        control_source = ColumnDataSource(
            data=dict(
                x=control_x,
                y=control_y,
                size=control_size,
                count=control_count
            )
        )
    else:
        control_source = None

    return ((treatment_source, control_source), (xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan))


def compute_histograms(xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan):
    # Create bins based on combined data-set
    h_bins = len(set(list(xs_treatment_nonan) + list(xs_control_nonan)))
    h_bins = min(h_bins, 10)
    y_bins = len(set(list(ys_treatment_nonan) + list(ys_control_nonan)))
    y_bins = min(y_bins, 20)
    y_bins = 20
    _, hedges = np.histogram(list(xs_treatment_nonan) + list(xs_control_nonan), bins=h_bins)
    _, vedges = np.histogram(list(ys_treatment_nonan) + list(ys_control_nonan), bins=y_bins)

    # create horizontal histogram - treatment
    hhist1, hedges1 = np.histogram(xs_treatment_nonan, bins=hedges)
    # Normalize count to percentage
    if (xs_treatment_nonan.size):
        hhist1 = [xx / xs_treatment_nonan.size * 100 for xx in hhist1]

    hzeros1 = np.zeros(len(hedges1) - 1)
    hmax1 = max(hhist1) * 1.2

    # create horizontal histogram - control
    bins = 10 if not xs_treatment_nonan.size else hedges1
    hhist2, hedges2 = np.histogram(xs_control_nonan, bins=hedges)

    # Normalize count to percentage
    if (xs_control_nonan.size):
        hhist2 = [xx / xs_control_nonan.size * 100 for xx in hhist2]

    hzeros2 = np.zeros(len(hedges2) - 1)
    hmax2 = max(hhist2) * 1.2

    # create vertical histogram - treatment
    vhist1, vedges1 = np.histogram(ys_treatment_nonan, bins=vedges)

    # Normalize count to percentage
    if (ys_treatment_nonan.size):
        vhist1 = [xx / ys_treatment_nonan.size * 100 for xx in vhist1]

    vzeros1 = np.zeros(len(vedges1) - 1)
    vmax1 = max(vhist1) * 1.2

    # create vertical histogram - control
    bins = 20 if not ys_treatment_nonan.size else vedges1
    vhist2, vedges2 = np.histogram(ys_control_nonan, bins=vedges)

    # Normalize count to percentage
    if (ys_control_nonan.size):
        vhist2 = [xx / ys_control_nonan.size * 100 for xx in vhist2]

    vzeros2 = np.zeros(len(vedges2) - 1)
    vmax2 = max(vhist2) * 1.2

    hmax = max(hmax1, hmax2)
    vmax = max(vmax1, vmax2)
    return ((hhist1, hedges1, hzeros1), (hhist2, hedges2, hzeros2), (vhist1, vedges1, vzeros1), (vhist2, vedges2, vzeros2), (hmax, vmax))


def create_figures():
    x_title = x.value
    y_title = y.value

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = Figure(plot_height=600, plot_width=800, tools=TOOLS, name='scatter', **kw)
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False

    if control_datasource is not None:
        p.circle(x='x', y='y', color=color_c, size='size', source=control_datasource, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5, legend="Control")

    if treatment_datasource is not None:
        p.circle(x='x', y='y', color=color_t, size='size', source=treatment_datasource, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5, legend="Intervention")

    if x.value == "mTICI Reperfusion score":
        p.xaxis.formatter = FuncTickFormatter(code="""
            if (tick==5){
                return "3";
            } else if (tick==4){
                return "2b";
            } else if (tick==3){
                return "2a";
            } else if (tick==2){
                return "1";
            } else if (tick==1){
                return "0";
            } else if (tick==0)
            {
                return "UTD";
            } else {
                return tick;
            }
            """)

    if y.value == "mTICI Reperfusion score":
        p.yaxis.formatter = FuncTickFormatter(code="""
            if (tick==5){
                return "3";
            } else if (tick==4){
                return "2b";
            } else if (tick==3){
                return "2a";
            } else if (tick==2){
                return "1";
            } else if (tick==1){
                return "0";
            } else if (tick==0)
            {
                return "UTD";
            } else {
                return tick;
            }
            """)

    LINE_ARGS = dict(color="#3A5785", line_color=None)

    ph1 = Figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
                 y_range=(-5, hmax), min_border=10, min_border_left=50, y_axis_location="right", name='hhist1')
    ph1.xgrid.grid_line_color = None
    ph1.yaxis.major_label_orientation = np.pi / 4
    ph1.background_fill_color = "#fafafa"
    ph1.yaxis.axis_label = '%'

    if ys_treatment_nonan.size and xs_treatment_nonan.size:
        ph1.quad(bottom=0, left=hedges1[:-1], right=hedges1[1:], top=hhist1, color=color_t, line_color="#3A5785")
        bin_width = hedges1[1] - hedges1[0]
        ph1.text([ xx + bin_width / 2 for xx in hedges1[:-1]],[hh + 0 for hh in hhist1], text = ["{:.1f}%".format(val) for val in hhist1], text_font_size="13pt", text_baseline="bottom", text_align="center")
        hht_1 = ph1.quad(bottom=0, left=hedges1[:-1], right=hedges1[1:], top=hzeros1, alpha=0.5, **LINE_ARGS)
        hht_2 = ph1.quad(bottom=0, left=hedges1[:-1], right=hedges1[1:], top=hzeros1, alpha=0.1, **LINE_ARGS)
    else:
        hht_1 = None
        hht_2 = None

    LINE_ARGS = dict(color="#3A5785", line_color=None)

    ph2 = Figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
                 y_range=(-5, hmax), min_border=10, min_border_left=50, y_axis_location="right", name='hhist2')
    ph2.xgrid.grid_line_color = None
    ph2.yaxis.major_label_orientation = np.pi / 4
    ph2.background_fill_color = "#fafafa"
    ph2.yaxis.axis_label = '%'

    if ys_control_nonan.size and xs_control_nonan.size:
        ph2.quad(bottom=0, left=hedges2[:-1], right=hedges2[1:], top=hhist2, color=color_c, line_color="#3A5785")
        bin_width = hedges2[1] - hedges2[0]
        ph2.text([ xx + bin_width / 2 for xx in hedges2[:-1]],[hh + 0 for hh in hhist2], text = ["{:.1f}%".format(val) for val in hhist2], text_font_size="13pt", text_baseline="bottom", text_align="center")
        hhc_1 = ph2.quad(bottom=0, left=hedges1[:-1], right=hedges1[1:], top=hzeros2, alpha=0.5, **LINE_ARGS)
        hhc_2 = ph2.quad(bottom=0, left=hedges1[:-1], right=hedges1[1:], top=hzeros2, alpha=0.1, **LINE_ARGS)
    else:
        hhc_1 = None
        hhc_2 = None

    pv1 = Figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height, x_range=(-10, vmax),
                 y_range=p.y_range, min_border=10, y_axis_location="right", name='vhist')
    pv1.ygrid.grid_line_color = None
    pv1.xaxis.major_label_orientation = np.pi / 4
    pv1.background_fill_color = "#fafafa"
    pv1.xaxis.axis_label = '%'

    if ys_treatment_nonan.size and xs_treatment_nonan.size:
        pv1.quad(left=0, bottom=vedges1[:-1], top=vedges1[1:], right=vhist1, color=color_t, line_color="#3A5785")
        bin_width = vedges1[1] - vedges1[0]
        pv1.text([ xx for xx in vhist1],[yy + bin_width / 2 for yy in vedges1[:-1]], text = [("{:.1f}%".format(val) if val != 0 else "") for val in vhist1], text_font_size="13pt", text_baseline="middle", text_align="left")
        vht_1 = pv1.quad(left=0, bottom=vedges1[:-1], top=vedges1[1:], right=vzeros1, alpha=0.5, **LINE_ARGS)
        vht_2 = pv1.quad(left=0, bottom=vedges1[:-1], top=vedges1[1:], right=vzeros1, alpha=0.1, **LINE_ARGS)
    else:
        vht_1 = None
        vht_2 = None

    pv2 = Figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height, x_range=(-10, vmax),
                 y_range=p.y_range, min_border=10, y_axis_location="right", name='vhist')
    pv2.ygrid.grid_line_color = None
    pv2.xaxis.major_label_orientation = np.pi / 4
    pv2.background_fill_color = "#fafafa"
    pv2.xaxis.axis_label = '%'

    if ys_control_nonan.size and xs_control_nonan.size:
        pv2.quad(left=0, bottom=vedges2[:-1], top=vedges2[1:], right=vhist2, color=color_c, line_color="#3A5785")
        bin_width = vedges2[1] - vedges2[0]
        pv2.text([ xx for xx in vhist2],[yy + bin_width / 2 for yy in vedges2[:-1]], text = [("{:.1f}%".format(val) if val != 0 else "") for val in vhist2], text_font_size="13pt", text_baseline="middle", text_align="left")
        vhc_1 = pv2.quad(left=0, bottom=vedges2[:-1], top=vedges2[1:], right=vzeros2, alpha=0.5, **LINE_ARGS)
        vhc_2 = pv2.quad(left=0, bottom=vedges2[:-1], top=vedges2[1:], right=vzeros2, alpha=0.1, **LINE_ARGS)
    else:
        vhc_1 = None
        vhc_2 = None

    return (p, ph1, ph2, pv1, pv2, (hht_1, hht_2, hhc_1, hhc_2, vht_1, vht_2, vhc_1, vhc_2))


def map_treatment_to_string(x):
    if (x == 1):
        return 'treatment'
    elif (x == 0):
        return 'control'
    else:
        return 'unknown'


def map_colors(x):
    if x['treatment'] is 1:
        return color_t
    else:
        return color_c


def update(attr, old, new):
    # scatter, hhist, vhist = layout.children[0].children[1], layout.children[1].children[1], layout.children[0].children[2]
    # (layout.children[0].children[1], layout.children[1].children[1], layout.children[0].children[2]) = create_figures()
    # (scatter, hhist, vhist) = create_figures()
    # rootLayout = curdoc().get_model_by_name('root')
    # listOfSublayouts = rootLayout.children
    # oldscatter = curdoc().get_model_by_name('scatter')
    # oldvhist = curdoc().get_model_by_name('vhist')
    # oldhhist = curdoc().get_model_by_name('hhist')

    # row1 = curdoc().get_model_by_name('row1')
    # row2 = curdoc().get_model_by_name('row2')
    # listOfSublayouts.remove(row1)
    # listOfSublayouts.remove(row2)
    # listOfSublayouts.append(row(controls, scatter, vhist, name='row1'))
    # listOfSublayouts.append(row(Spacer(width=200), hhist, name='row2'))
    #
    # curdoc().clear()
    # curdoc().remove(rootLayout)
    # newLayout = column(row(controls, scatter, vhist), row(Spacer(width=200), hhist))
    # curdoc().add_root(newLayout)

    # (scatter, hhist, vhist) = create_figures()
    # layout.children[1].children[0] = scatter
    # layout.children[1].children[1] = hhist
    # layout.children[2] = vhist
    global treatment_datasource, control_datasource, xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan
    global scatter, hhist_t, hhist_c, vhist_t, vhist_c
    global hht_1, hht_2, hhc_1, hhc_2, vht_1, vht_2, vhc_1, vhc_2
    global hhist1, hedges1, hzeros1, hhist2, hedges2, hzeros2, vhist1, vedges1, vzeros1, vhist2, vedges2, vzeros2, hmax, vmax

    ((treatment_datasource, control_datasource), (xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan)) = create_data_sources()
    if treatment_datasource is not None:
        treatment_datasource.on_change('selected', update_treatment_selected)
    if control_datasource is not None:
        control_datasource.on_change('selected', update_control_selected)

    ((hhist1, hedges1, hzeros1), (hhist2, hedges2, hzeros2),
     (vhist1, vedges1, vzeros1), (vhist2, vedges2, vzeros2),
     (hmax, vmax)) = compute_histograms(xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan)

    (scatter, hhist_t, hhist_c, vhist_t, vhist_c, selection_overlays) = create_figures()
    (hht_1, hht_2, hhc_1, hhc_2, vht_1, vht_2, vhc_1, vhc_2) = selection_overlays
    # layoutDef = layout([controls, scatter, vhist], [Spacer(width=200), hhist], sizing_mode='fixed')
    layoutDef = row([column([controls]), column([scatter, hhist_t, hhist_c]), column([vhist_t]), column([vhist_c])], sizing_mode='fixed')
    layoutDef.name = 'subroot'

    rootLayout = curdoc().get_model_by_name('root')
    listOfSublayouts = rootLayout.children
    toRemove = curdoc().get_model_by_name('subroot')
    listOfSublayouts.remove(toRemove)
    listOfSublayouts.append(layoutDef)
    # (layoutDef.children[0].children[1], layoutDef.children[1].children[1], layoutDef.children[0].children[2]) = create_figures()


def update_treatment_selected(attr, old, new):
    inds = np.array(treatment_datasource.selected['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(treatment_datasource.data['x']):
        hhist1, hhist2 = hzeros1, hzeros1
        vhist1, vhist2 = vzeros1, vzeros1
    else:
        neg_inds = np.ones_like(treatment_datasource.data['x'], dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(np.array(treatment_datasource.data['x'])[inds], weights=np.array(treatment_datasource.data['count'])[inds], bins=hedges1)
        hhist1 = [ii / len(xs_treatment_nonan) * 100 for ii in hhist1]
        vhist1, _ = np.histogram(np.array(treatment_datasource.data['y'])[inds], weights=np.array(treatment_datasource.data['count'])[inds], bins=vedges1)
        vhist1 = [ii / len(ys_treatment_nonan) * 100 for ii in vhist1]
        hhist2, _ = np.histogram(np.array(treatment_datasource.data['x'])[neg_inds], weights=np.array(treatment_datasource.data['count'])[neg_inds], bins=hedges1)
        hhist2 = [ii / len(xs_treatment_nonan) * 100 for ii in hhist2]
        vhist2, _ = np.histogram(np.array(treatment_datasource.data['y'])[neg_inds], weights=np.array(treatment_datasource.data['count'])[neg_inds], bins=vedges1)
        vhist2 = [ii / len(ys_treatment_nonan) * 100 for ii in vhist2]

    hht_1.data_source.data["top"] = hhist1
    hht_2.data_source.data["top"] = -np.array(hhist2)
    vht_1.data_source.data["right"] = vhist1
    vht_2.data_source.data["right"] = -np.array(vhist2)


def update_control_selected(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(control_datasource.data['x']):
        hhist1, hhist2 = hzeros2, hzeros2
        vhist1, vhist2 = vzeros2, vzeros2
    else:
        neg_inds = np.ones_like(control_datasource.data['x'], dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(np.array(control_datasource.data['x'])[inds], weights=np.array(control_datasource.data['count'])[inds], bins=hedges2)
        hhist1 = [ii / len(xs_control_nonan) * 100 for ii in hhist1]
        vhist1, _ = np.histogram(np.array(control_datasource.data['y'])[inds], weights=np.array(control_datasource.data['count'])[inds], bins=vedges2)
        vhist1 = [ii / len(ys_control_nonan) * 100 for ii in vhist1]
        hhist2, _ = np.histogram(np.array(control_datasource.data['x'])[neg_inds], weights=np.array(control_datasource.data['count'])[neg_inds], bins=hedges2)
        hhist2 = [ii / len(xs_control_nonan) * 100 for ii in hhist2]
        vhist2, _ = np.histogram(np.array(control_datasource.data['y'])[neg_inds], weights=np.array(control_datasource.data['count'])[neg_inds], bins=vedges2)
        vhist2 = [ii / len(ys_control_nonan) * 100 for ii in vhist2]

    hhc_1.data_source.data["top"] = hhist1
    hhc_2.data_source.data["top"] = -np.array(hhist2)
    vhc_1.data_source.data["right"] = vhist1
    vhc_2.data_source.data["right"] = -np.array(vhist2)


x = Select(title='X-Axis', value='Baseline NIHSS', options=list(xnames.keys()))
x.on_change('value', update)

y = Select(title='Y-Axis', value='MRS-Day 90', options=list(ynames.keys()))
y.on_change('value', update)

# size = Select(title='Size', value='None', options=['None'] + list(set(list(xnames.keys()) + list(ynames.keys()))))
# size.on_change('value', update)

# Colour is used for treatment vs control
# color = Select(title='Color', value='mrs_D90', options=['None'] + columns)
# color.on_change('value', update)

treatment_title = Paragraph(text="Treatment Group")
treatment = RadioButtonGroup(labels=['Intervention', 'Control', 'Both'], active=2)
treatment.on_change('active', update)
sex_title = Paragraph(text="Sex")
sex = RadioButtonGroup(labels=['Male', 'Female', 'Both'], active=2)
sex.on_change('active', update)
# labels=['None', 'Diabetes', 'Smoking', 'Blood Thinners', 'Atrial fibrillation', 'Cancer', 'Sepsis', 'Pneumonia']
conditions_title = Paragraph(text="Patient History")
# conditions = CheckboxButtonGroup(labels=['None'] + [condition_names[cond] for cond in pre_conditions], active=list(range(0, len(pre_conditions) + 1)))
# conditions.on_change('active', update)
conditions_controls = [RadioButtonGroup(labels=[condition_names[cond] + ' (Either)', 'Yes', 'No'], active=0) for cond in pre_conditions]
# Set all on_change functions
[control.on_change('active', update) for control in conditions_controls]

# controls = widgetbox([x, y, color, size, treatment, conditions], width=200)
# controls = widgetbox([x, y, size, treatment_title, treatment, sex_title, sex, conditions_title, conditions], width=250)
controls = widgetbox([x, y, treatment_title, treatment, sex_title, sex, conditions_title] + conditions_controls, width=300)

((treatment_datasource, control_datasource), (xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan)) = create_data_sources()
((hhist1, hedges1, hzeros1), (hhist2, hedges2, hzeros2),
 (vhist1, vedges1, vzeros1), (vhist2, vedges2, vzeros2),
 (hmax, vmax)) = compute_histograms(xs_treatment_nonan, xs_control_nonan, ys_treatment_nonan, ys_control_nonan)

treatment_datasource.on_change('selected', update_treatment_selected)
control_datasource.on_change('selected', update_control_selected)

(scatter, hhist_t, hhist_c, vhist_t, vhist_c, selection_overlays) = create_figures()
(hht_1, hht_2, hhc_1, hhc_2, vht_1, vht_2, vhc_1, vhc_2) = selection_overlays

# layout = column(row(controls, scatter, vhist, name='row1'), row(Spacer(width=200), hhist, name='row2'), name='root')
# layout1 = Row(controls, scatter, vhist)
# layout2 = Row(Spacer(width=200), hhist)

# layoutDef = Row(controls, gridplot([scatter, vhist], [hhist]))

layoutDef = row([column([controls]), column([scatter, hhist_t, hhist_c]), column([vhist_t]), column([vhist_c])], sizing_mode='fixed')
layoutDef.name = 'subroot'
rootLayout = Column(layoutDef, name='root')

curdoc().add_root(rootLayout)
curdoc().title = "Crossfilter"