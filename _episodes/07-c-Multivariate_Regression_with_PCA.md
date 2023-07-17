---
title: Feature selection with PCA
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- ""
questions:
- ""
---

# High Dimensional Data Analysis Day 1

<!-- from traitlets.config.manager import BaseJSONConfigManager
from pathlib import Path
path = Path.home() / ".jupyter" / "nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "theme": "black",
        "transition": None,
        "start_slideshow_at": "selected",
        "enable_chalkboard": True,
        "chalkboard": {
            "color": ["rgb(225, 193, 7)", "rgb(30, 136, 229)"]
        },
     }
) -->


```python
# limit # of obs.
# train = 350
# test = 350
#kfold splits
n_splits = 3

# try onehot encoding nom vars (see Chris's helper function)
# comparison with less and more data.
```

Motivate this. Context for disucssion high dim analysis.

# Predict if house sales price will be high for market from house characteristics

## Ames housing dataset data


## load dataset


```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
```

## view data


```python
df = housing.data.copy(deep=True)
df = df.astype({'Id':int})  # set data type of Id to int
df = df.set_index('Id')  # set Id column to be the index of the DataFrame
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60.0</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2007.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70.0</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2006.0</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60.0</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>60.0</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>2007.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>20.0</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>70.0</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500.0</td>
      <td>5.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>20.0</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>20.0</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 79 columns</p>
</div>



## all feature names


```python
print(df.columns.tolist())
```

    ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']


# Reminder to access the Data Dictionary


```python
from IPython.display import display, Pretty

display(Pretty(housing.DESCR))

```


    Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

    With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

    MSSubClass: Identifies the type of dwelling involved in the sale.

            20	1-STORY 1946 & NEWER ALL STYLES
            30	1-STORY 1945 & OLDER
            40	1-STORY W/FINISHED ATTIC ALL AGES
            45	1-1/2 STORY - UNFINISHED ALL AGES
            50	1-1/2 STORY FINISHED ALL AGES
            60	2-STORY 1946 & NEWER
            70	2-STORY 1945 & OLDER
            75	2-1/2 STORY ALL AGES
            80	SPLIT OR MULTI-LEVEL
            85	SPLIT FOYER
            90	DUPLEX - ALL STYLES AND AGES
           120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
           150	1-1/2 STORY PUD - ALL AGES
           160	2-STORY PUD - 1946 & NEWER
           180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
           190	2 FAMILY CONVERSION - ALL STYLES AND AGES

    MSZoning: Identifies the general zoning classification of the sale.

           A	Agriculture
           C	Commercial
           FV	Floating Village Residential
           I	Industrial
           RH	Residential High Density
           RL	Residential Low Density
           RP	Residential Low Density Park
           RM	Residential Medium Density

    LotFrontage: Linear feet of street connected to property

    LotArea: Lot size in square feet

    Street: Type of road access to property

           Grvl	Gravel
           Pave	Paved

    Alley: Type of alley access to property

           Grvl	Gravel
           Pave	Paved
           NA 	No alley access

    LotShape: General shape of property

           Reg	Regular
           IR1	Slightly irregular
           IR2	Moderately Irregular
           IR3	Irregular

    LandContour: Flatness of the property

           Lvl	Near Flat/Level
           Bnk	Banked - Quick and significant rise from street grade to building
           HLS	Hillside - Significant slope from side to side
           Low	Depression

    Utilities: Type of utilities available

           AllPub	All public Utilities (E,G,W,& S)
           NoSewr	Electricity, Gas, and Water (Septic Tank)
           NoSeWa	Electricity and Gas Only
           ELO	Electricity only

    LotConfig: Lot configuration

           Inside	Inside lot
           Corner	Corner lot
           CulDSac	Cul-de-sac
           FR2	Frontage on 2 sides of property
           FR3	Frontage on 3 sides of property

    LandSlope: Slope of property

           Gtl	Gentle slope
           Mod	Moderate Slope
           Sev	Severe Slope

    Neighborhood: Physical locations within Ames city limits

           Blmngtn	Bloomington Heights
           Blueste	Bluestem
           BrDale	Briardale
           BrkSide	Brookside
           ClearCr	Clear Creek
           CollgCr	College Creek
           Crawfor	Crawford
           Edwards	Edwards
           Gilbert	Gilbert
           IDOTRR	Iowa DOT and Rail Road
           MeadowV	Meadow Village
           Mitchel	Mitchell
           Names	North Ames
           NoRidge	Northridge
           NPkVill	Northpark Villa
           NridgHt	Northridge Heights
           NWAmes	Northwest Ames
           OldTown	Old Town
           SWISU	South & West of Iowa State University
           Sawyer	Sawyer
           SawyerW	Sawyer West
           Somerst	Somerset
           StoneBr	Stone Brook
           Timber	Timberland
           Veenker	Veenker

    Condition1: Proximity to various conditions

           Artery	Adjacent to arterial street
           Feedr	Adjacent to feeder street
           Norm	Normal
           RRNn	Within 200' of North-South Railroad
           RRAn	Adjacent to North-South Railroad
           PosN	Near positive off-site feature--park, greenbelt, etc.
           PosA	Adjacent to postive off-site feature
           RRNe	Within 200' of East-West Railroad
           RRAe	Adjacent to East-West Railroad

    Condition2: Proximity to various conditions (if more than one is present)

           Artery	Adjacent to arterial street
           Feedr	Adjacent to feeder street
           Norm	Normal
           RRNn	Within 200' of North-South Railroad
           RRAn	Adjacent to North-South Railroad
           PosN	Near positive off-site feature--park, greenbelt, etc.
           PosA	Adjacent to postive off-site feature
           RRNe	Within 200' of East-West Railroad
           RRAe	Adjacent to East-West Railroad

    BldgType: Type of dwelling

           1Fam	Single-family Detached
           2FmCon	Two-family Conversion; originally built as one-family dwelling
           Duplx	Duplex
           TwnhsE	Townhouse End Unit
           TwnhsI	Townhouse Inside Unit

    HouseStyle: Style of dwelling

           1Story	One story
           1.5Fin	One and one-half story: 2nd level finished
           1.5Unf	One and one-half story: 2nd level unfinished
           2Story	Two story
           2.5Fin	Two and one-half story: 2nd level finished
           2.5Unf	Two and one-half story: 2nd level unfinished
           SFoyer	Split Foyer
           SLvl	Split Level

    OverallQual: Rates the overall material and finish of the house

           10	Very Excellent
           9	Excellent
           8	Very Good
           7	Good
           6	Above Average
           5	Average
           4	Below Average
           3	Fair
           2	Poor
           1	Very Poor

    OverallCond: Rates the overall condition of the house

           10	Very Excellent
           9	Excellent
           8	Very Good
           7	Good
           6	Above Average
           5	Average
           4	Below Average
           3	Fair
           2	Poor
           1	Very Poor

    YearBuilt: Original construction date

    YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

    RoofStyle: Type of roof

           Flat	Flat
           Gable	Gable
           Gambrel	Gabrel (Barn)
           Hip	Hip
           Mansard	Mansard
           Shed	Shed

    RoofMatl: Roof material

           ClyTile	Clay or Tile
           CompShg	Standard (Composite) Shingle
           Membran	Membrane
           Metal	Metal
           Roll	Roll
           Tar&Grv	Gravel & Tar
           WdShake	Wood Shakes
           WdShngl	Wood Shingles

    Exterior1st: Exterior covering on house

           AsbShng	Asbestos Shingles
           AsphShn	Asphalt Shingles
           BrkComm	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           CemntBd	Cement Board
           HdBoard	Hard Board
           ImStucc	Imitation Stucco
           MetalSd	Metal Siding
           Other	Other
           Plywood	Plywood
           PreCast	PreCast
           Stone	Stone
           Stucco	Stucco
           VinylSd	Vinyl Siding
           Wd Sdng	Wood Siding
           WdShing	Wood Shingles

    Exterior2nd: Exterior covering on house (if more than one material)

           AsbShng	Asbestos Shingles
           AsphShn	Asphalt Shingles
           BrkComm	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           CemntBd	Cement Board
           HdBoard	Hard Board
           ImStucc	Imitation Stucco
           MetalSd	Metal Siding
           Other	Other
           Plywood	Plywood
           PreCast	PreCast
           Stone	Stone
           Stucco	Stucco
           VinylSd	Vinyl Siding
           Wd Sdng	Wood Siding
           WdShing	Wood Shingles

    MasVnrType: Masonry veneer type

           BrkCmn	Brick Common
           BrkFace	Brick Face
           CBlock	Cinder Block
           None	None
           Stone	Stone

    MasVnrArea: Masonry veneer area in square feet

    ExterQual: Evaluates the quality of the material on the exterior

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    ExterCond: Evaluates the present condition of the material on the exterior

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    Foundation: Type of foundation

           BrkTil	Brick & Tile
           CBlock	Cinder Block
           PConc	Poured Contrete
           Slab	Slab
           Stone	Stone
           Wood	Wood

    BsmtQual: Evaluates the height of the basement

           Ex	Excellent (100+ inches)
           Gd	Good (90-99 inches)
           TA	Typical (80-89 inches)
           Fa	Fair (70-79 inches)
           Po	Poor (<70 inches
           NA	No Basement

    BsmtCond: Evaluates the general condition of the basement

           Ex	Excellent
           Gd	Good
           TA	Typical - slight dampness allowed
           Fa	Fair - dampness or some cracking or settling
           Po	Poor - Severe cracking, settling, or wetness
           NA	No Basement

    BsmtExposure: Refers to walkout or garden level walls

           Gd	Good Exposure
           Av	Average Exposure (split levels or foyers typically score average or above)
           Mn	Mimimum Exposure
           No	No Exposure
           NA	No Basement

    BsmtFinType1: Rating of basement finished area

           GLQ	Good Living Quarters
           ALQ	Average Living Quarters
           BLQ	Below Average Living Quarters
           Rec	Average Rec Room
           LwQ	Low Quality
           Unf	Unfinshed
           NA	No Basement

    BsmtFinSF1: Type 1 finished square feet

    BsmtFinType2: Rating of basement finished area (if multiple types)

           GLQ	Good Living Quarters
           ALQ	Average Living Quarters
           BLQ	Below Average Living Quarters
           Rec	Average Rec Room
           LwQ	Low Quality
           Unf	Unfinshed
           NA	No Basement

    BsmtFinSF2: Type 2 finished square feet

    BsmtUnfSF: Unfinished square feet of basement area

    TotalBsmtSF: Total square feet of basement area

    Heating: Type of heating

           Floor	Floor Furnace
           GasA	Gas forced warm air furnace
           GasW	Gas hot water or steam heat
           Grav	Gravity furnace
           OthW	Hot water or steam heat other than gas
           Wall	Wall furnace

    HeatingQC: Heating quality and condition

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           Po	Poor

    CentralAir: Central air conditioning

           N	No
           Y	Yes

    Electrical: Electrical system

           SBrkr	Standard Circuit Breakers & Romex
           FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)
           FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
           FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
           Mix	Mixed

    1stFlrSF: First Floor square feet

    2ndFlrSF: Second floor square feet

    LowQualFinSF: Low quality finished square feet (all floors)

    GrLivArea: Above grade (ground) living area square feet

    BsmtFullBath: Basement full bathrooms

    BsmtHalfBath: Basement half bathrooms

    FullBath: Full bathrooms above grade

    HalfBath: Half baths above grade

    Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

    Kitchen: Kitchens above grade

    KitchenQual: Kitchen quality

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor

    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

    Functional: Home functionality (Assume typical unless deductions are warranted)

           Typ	Typical Functionality
           Min1	Minor Deductions 1
           Min2	Minor Deductions 2
           Mod	Moderate Deductions
           Maj1	Major Deductions 1
           Maj2	Major Deductions 2
           Sev	Severely Damaged
           Sal	Salvage only

    Fireplaces: Number of fireplaces

    FireplaceQu: Fireplace quality

           Ex	Excellent - Exceptional Masonry Fireplace
           Gd	Good - Masonry Fireplace in main level
           TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
           Fa	Fair - Prefabricated Fireplace in basement
           Po	Poor - Ben Franklin Stove
           NA	No Fireplace

    GarageType: Garage location

           2Types	More than one type of garage
           Attchd	Attached to home
           Basment	Basement Garage
           BuiltIn	Built-In (Garage part of house - typically has room above garage)
           CarPort	Car Port
           Detchd	Detached from home
           NA	No Garage

    GarageYrBlt: Year garage was built

    GarageFinish: Interior finish of the garage

           Fin	Finished
           RFn	Rough Finished
           Unf	Unfinished
           NA	No Garage

    GarageCars: Size of garage in car capacity

    GarageArea: Size of garage in square feet

    GarageQual: Garage quality

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor
           NA	No Garage

    GarageCond: Garage condition

           Ex	Excellent
           Gd	Good
           TA	Typical/Average
           Fa	Fair
           Po	Poor
           NA	No Garage

    PavedDrive: Paved driveway

           Y	Paved
           P	Partial Pavement
           N	Dirt/Gravel

    WoodDeckSF: Wood deck area in square feet

    OpenPorchSF: Open porch area in square feet

    EnclosedPorch: Enclosed porch area in square feet

    3SsnPorch: Three season porch area in square feet

    ScreenPorch: Screen porch area in square feet

    PoolArea: Pool area in square feet

    PoolQC: Pool quality

           Ex	Excellent
           Gd	Good
           TA	Average/Typical
           Fa	Fair
           NA	No Pool

    Fence: Fence quality

           GdPrv	Good Privacy
           MnPrv	Minimum Privacy
           GdWo	Good Wood
           MnWw	Minimum Wood/Wire
           NA	No Fence

    MiscFeature: Miscellaneous feature not covered in other categories

           Elev	Elevator
           Gar2	2nd Garage (if not described in garage section)
           Othr	Other
           Shed	Shed (over 100 SF)
           TenC	Tennis Court
           NA	None

    MiscVal: $Value of miscellaneous feature

    MoSold: Month Sold (MM)

    YrSold: Year Sold (YYYY)

    SaleType: Type of sale

           WD 	Warranty Deed - Conventional
           CWD	Warranty Deed - Cash
           VWD	Warranty Deed - VA Loan
           New	Home just constructed and sold
           COD	Court Officer Deed/Estate
           Con	Contract 15% Down payment regular terms
           ConLw	Contract Low Down payment and low interest
           ConLI	Contract Low Interest
           ConLD	Contract Low Down
           Oth	Other

    SaleCondition: Condition of sale

           Normal	Normal Sale
           Abnorml	Abnormal Sale -  trade, foreclosure, short sale
           AdjLand	Adjoining Land Purchase
           Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit
           Family	Sale between family members
           Partial	Home was not completed when last assessed (associated with New Homes)

    Downloaded from openml.org.


#### EXERCISE_START
What does TotRmsAbvGrd refer to?
> > ## Solution
> >
> > TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
> {:.solution}
{:.challenge}


#### EXERCISE_START
How many variables are numeric?
> > ## Solution
> >
> > 36 numeric (are these all continuous or discrete?)
> > 43 categorical (are these all categorical or ordinate ?)
> {:.solution}
{:.challenge}


#### EXERCISE_START
How many Nan entries are there per variable?
> > ## Solution
> >
> > ~~~
> > df.isna().sum()
> > ~~~
> > {: .language-python}
> {:.solution}
{:.challenge}


#### EXERCISE_START
Which of these variables could be the best predictor of house sale price? Why?
> > ## Solution
> >
> > Possible answers: SquareFt, OverallQual, YearBuilt
> > They intutively are going to be corrleated with SalePrice - but NB: also with each other!
> {:.solution}
{:.challenge}



# Target Feature: SalePrice


```python
# add target variable 'sales price' to data df from housing object
df[housing.target_names[0]] = housing.target.tolist()
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>



## what does SalePrice look like?


```python
import helper_functions
helper_functions.plot_salesprice(
    df,
    #ylog=True
)
```







Is this a normal distribution? Will that distribution influcence modelling this value? How?

# Feature Selection


```python
# Original DataFrame dimensions (+ SalesPrice)
print(f"{df.shape=}")
```

    df.shape=(1460, 80)



```python
# create dummy variables/ one hot encode dummy variables
import pandas as pd
numeric_variables = df.describe().columns.tolist()
nominative_variables = [x for x in df.columns.tolist() if x not in numeric_variables]

dummy_df = pd.get_dummies(df[nominative_variables])
print(dummy_df.shape)
dummy_df
```

    (1460, 252)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning_C (all)</th>
      <th>MSZoning_FV</th>
      <th>MSZoning_RH</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>Street_Grvl</th>
      <th>Street_Pave</th>
      <th>Alley_Grvl</th>
      <th>Alley_Pave</th>
      <th>LotShape_IR1</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 252 columns</p>
</div>




```python
model_df = pd.concat([df[numeric_variables], dummy_df], axis=1) #.drop('SalePrice', axis=1)
print(model_df.shape)
model_df
```

    (1460, 289)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>60</td>
      <td>62.0</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>20</td>
      <td>85.0</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>119.0</td>
      <td>790</td>
      <td>163</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>70</td>
      <td>66.0</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>0.0</td>
      <td>275</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>20</td>
      <td>68.0</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>0.0</td>
      <td>49</td>
      <td>1029</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>20</td>
      <td>75.0</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>0.0</td>
      <td>830</td>
      <td>290</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 289 columns</p>
</div>




```python
# for simplicity at this piont - let's only use numerical columns
# only numerical column descriptions
model_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>...</td>
      <td>0.003425</td>
      <td>0.083562</td>
      <td>0.002055</td>
      <td>0.867808</td>
      <td>0.069178</td>
      <td>0.002740</td>
      <td>0.008219</td>
      <td>0.013699</td>
      <td>0.820548</td>
      <td>0.085616</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>...</td>
      <td>0.058440</td>
      <td>0.276824</td>
      <td>0.045299</td>
      <td>0.338815</td>
      <td>0.253844</td>
      <td>0.052289</td>
      <td>0.090317</td>
      <td>0.116277</td>
      <td>0.383862</td>
      <td>0.279893</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 289 columns</p>
</div>



#### EXERCISE_START
#### Modelling Dataset Description
1. how many observations are there in our dataset?
2. how many features are there in the whole dataset?
3. how many **numerical** features are there in the whole dataset?

> > ## Solution
> >
> > 1. 1460 observations (len(df))
> > 2. 79 features total (len(df.columns.tolist())) - 1 (can't use SalesPrice)
> > 3. 36 numerical features (len(df[num_cols].columns.tolist()) - 1 (can't use SalesPrice)
> > 
> {:.solution}
{:.challenge}



> ## Modelling Feature Selection
> 1. Can all of those features be used in a model?
> 2. Would you want to use all of those features?
> > ## Solution
> >
> > 1. yes all the features could be used. With possible implications for the quality of the model.
> > 3. features that are not (anti)correlated with the target variable may not add any useful information to the model
> > 3. features that are correlated with other features may not add a lot more information and may produce a poorer quality model.
> > 
> {:.solution}
{:.challenge}


> ## Model Feature Count
> 2. how many features should be used total?
> > ## Solution
> >
> > ### A possible approach:
> > 0. n = number of observations
> > 1. uncorrelated features count = (n - 1)
> > 2. as correlation increases, feature count proportional to sqrt(n)
> > 1. assuming some correlation: sqrt(1460) = 38.21
> > per: [Optimal number of features as a function of sample size for various classification rules](https://academic.oup.com/bioinformatics/article/21/8/1509/249540)
> > 
> > ### Data analysis and modeling can be very emprical
> > You need to try things out to see what works. If your features are indepent and identically distributed, or not, will impact how many observations are required
> > 
> > ### Generally for a classifcation model
> > 1. Distribution of features per target class matters a ton
> > 2. More observations mean you can use more features
> {:.solution}
{:.challenge}



> ## Overfitting
> What is model overfitting? how does a model become overfit?
> 
> > ## Solution
> >
> > your model is unabel to generalize - it has 'memorized' the data, rather than the patterns in it.
> > 
> > TODO: ADD IN HERE.
> > 
> > ##### EXERCISE_END
> > 
> > 
> ## Model Feature Quality
> 4. which features should be used to predict the target variable? (which variables are good predictors?)
> > ## Solution
> >
> > Many possible answers here, some general ideas
> > 1. those that are most correlated with the target variable
> > 2. those that are not correlated with each other
> {:.solution}
{:.challenge}


# Build regression model to predict sales price

## Plot correlations and histograms of those columns

Reminder:
1. What features should go in a model to predict high house price?
2. What features are correlated with high house price?


```python
corr_mat=helper_functions.plot_corr_matrix_allVars(model_df)

```







## check top 5 highest correlation values


```python
corr_cols = (
    corr_mat['SalePrice']
    .sort_values(ascending=False)
    .index
    .tolist()
)
corr_cols[1:6]

```




    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']



# Create Pairplots of the top five most correlated values


```python
# Plot correlations and histograms of those columns, color with hue. This might take a while.
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(
    model_df[corr_cols[1:6] + ['SalePrice']],
    dropna=True,
    corner=True,
    plot_kws={'alpha': 0.3}
)

plt.show()
```







# Baseline - if always guess mean - RMSE?


```python
from math import sqrt
mean_sale_price = model_df.SalePrice.mean()
diffs = model_df.SalePrice - mean_sale_price
rse = (diffs * diffs).apply(sqrt)
baseline_rmse = rse.mean()
print('baseline rmse: {:2f}'.format(baseline_rmse))
```

    baseline rmse: 57434.770276


# Remove nulls from features


```python
# which columns have the most nulls
model_df.isnull().sum().sort_values(ascending=False).head(20)
```




    LotFrontage         259
    GarageYrBlt          81
    MasVnrArea            8
    MSSubClass            0
    BsmtExposure_Av       0
    BsmtFinType1_GLQ      0
    BsmtFinType1_BLQ      0
    BsmtFinType1_ALQ      0
    BsmtExposure_No       0
    BsmtExposure_Mn       0
    BsmtExposure_Gd       0
    BsmtCond_TA           0
    BsmtFinType1_Rec      0
    BsmtCond_Po           0
    BsmtCond_Gd           0
    BsmtCond_Fa           0
    BsmtQual_TA           0
    BsmtQual_Gd           0
    BsmtQual_Fa           0
    BsmtFinType1_LwQ      0
    dtype: int64




```python
# assume null means none - replace all nulls with zeros for lotFrontage and MasVnrArea
no_null_model_df = model_df
no_null_model_df['LotFrontage'] = no_null_model_df['LotFrontage'].fillna(0)
no_null_model_df['MasVnrArea'] = no_null_model_df['MasVnrArea'].fillna(0)

# GarageYrBlt 0 makes no sense - replace with mean
no_null_model_df['GarageYrBlt'] = no_null_model_df['GarageYrBlt'].fillna(no_null_model_df['GarageYrBlt'].mean())
no_null_model_df.isnull().sum().sort_values(ascending=False).head(20)
```




    MSSubClass             0
    Exterior1st_VinylSd    0
    BsmtFinType1_GLQ       0
    BsmtFinType1_BLQ       0
    BsmtFinType1_ALQ       0
    BsmtExposure_No        0
    BsmtExposure_Mn        0
    BsmtExposure_Gd        0
    BsmtExposure_Av        0
    BsmtCond_TA            0
    BsmtCond_Po            0
    BsmtCond_Gd            0
    BsmtCond_Fa            0
    BsmtQual_TA            0
    BsmtQual_Gd            0
    BsmtQual_Fa            0
    BsmtQual_Ex            0
    BsmtFinType1_LwQ       0
    BsmtFinType1_Rec       0
    BsmtFinType1_Unf       0
    dtype: int64



# separate features from target


```python
features = no_null_model_df.drop('SalePrice', axis=1)
features
target = no_null_model_df['SalePrice']
```


```python
# confirm features do not contain target
[x for x in features.columns if x == 'SalePrice']
```




    []



# Linear model:
## all non-null, numeric dimensions


```python
from collections import defaultdict
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple
from math import sqrt
import numpy as np


def run_linear_regression_with_kf(features: pd.DataFrame, labels: pd.Series,
                                    n_splits=5, title='logistic regression model'
                                   ) -> Tuple[float,float,float,float]:
    """
    scale, split, and model data. Return model performance statistics, plot confusion matrix
    feature: dataframe of feature columns to model
    labels: series of labels to model against
    test_size: fraction of labels to use in test split
    title: title for chart
    return: recall mean, recall sd, precision mean, precision sd
    """
    # set up splits/folds and array for stats.
    kf = StratifiedKFold(n_splits=n_splits)
    r2s = np.zeros(n_splits)
    rmses = np.zeros(n_splits)

    # fit model for each split/fold
    for i, (train_idx, test_idx) in enumerate(kf.split(X=features, y=labels)):
        # split data
        try:
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = labels.iloc[test_idx]

        except AttributeError:  # ndarray doesn't have .iloc
            X_train = features[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features[test_idx]
            y_test = labels.iloc[test_idx]


        # scale all features to training features
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fit model, evaluate
        regr = LinearRegression().fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        r2s[i] = r2_score(y_test, y_pred)
        rmses[i] = sqrt(mean_squared_error(y_test, y_pred))

    r2_mean = r2s.mean()
    r2_sd = r2s.std()
    rmse_mean = rmses.mean()
    rmse_sd = rmses.std()


    # plot mean confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.set_title(f'{title}\n' \
                 'mean r2: {:.2f},\n'\
                 'mean rmse {:.2f}'
                 .format(r2_mean, rmse_mean)
    )
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    plt.show()
    return r2_mean, rmse_mean, r2_sd, rmse_sd



```


```python
all_stats = {}
all_stats['all'] = run_linear_regression_with_kf(features=features, labels=target, n_splits=n_splits, title='all feats')
```

    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(









```python
corr_12 = (
    corr_mat['SalePrice']
    .sort_values(ascending=False)[1:13]
    .index
    .tolist()
)
print(features[corr_12].shape)
all_stats['top12'] = run_linear_regression_with_kf(features=features[corr_12], labels=target, n_splits=n_splits, title='Top 12 feats correlated with sale price')


```

    (1460, 12)


    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(








# Linear model:
## top 5 most correlated dimensions only


```python
corr_5 = (
    corr_mat['SalePrice']
    .sort_values(ascending=False)[1:6]
    .index
    .tolist()
)
print(features[corr_5].shape)
all_stats['top6'] = run_linear_regression_with_kf(features=features[corr_5], labels=target, n_splits=n_splits, title='Top 5 feats correlated with sale price')


```

    (1460, 5)


    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(









```python
# Linear model:
## top 12 most correlated dimensions only
```

# Linear Model:
## 2D PCA of all columns


```python
features.shape
```




    (1460, 288)




```python
from sklearn.decomposition import PCA
p = PCA(n_components=100)
features_pca = p.fit_transform(features)
all_stats['all_pca'] = run_linear_regression_with_kf(features=features_pca, labels=target,
                                                title='33 Princpal Components', n_splits=n_splits)

```

    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(









```python
from sklearn.decomposition import PCA
p = PCA(n_components=12)
features_pca_12 = p.fit_transform(features)

all_stats['12d_pca'] = run_linear_regression_with_kf(features=features_pca_12, labels=target,
                                                title='12 Princpal Components', n_splits=n_splits)

```

    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(









```python
from sklearn.decomposition import PCA
p = PCA(n_components=5)
features_pca_5 = p.fit_transform(features)

all_stats['5d_pca'] = run_linear_regression_with_kf(features=features_pca_5, labels=target,
                                                title='5 Princpal Components', n_splits=n_splits)

```

    C:\Users\Endemann\anaconda3\envs\highdim_workshop\Lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
      warnings.warn(








# Model Comparison


```python
# create combined stats df
stats_df = pd.DataFrame.from_dict(all_stats).set_index(
    pd.Index(['r2_mean', 'rmse_mean', 'r2_sd', 'rmse_sd'], name='statistics')
)

# plot figures
fig, axs = plt.subplots(1,2, figsize=(10, 5))
stats_df.loc['r2_mean'].plot(ax=axs[0], kind='bar', yerr=stats_df.loc['r2_sd'], title='mean r2',  color='lightblue', ylim=(-1,1))
stats_df.loc['rmse_mean'].plot(ax=axs[1], kind='bar', yerr=stats_df.loc['rmse_sd'], title=f'mean RMSE',  color='orange', ylim=(0, 100_000))
# plot baseline - guess mean every time RMSE
xmin, xmax = plt.xlim()
axs[1].hlines(baseline_rmse, xmin=xmin, xmax=xmax)
axs[1].text(xmax/3, baseline_rmse + 1000, 'Baseline RMSE')
plt.suptitle(f'model statistics\nerrbars=sd n={n_splits}')
plt.show()

```







> ## Why Didn't The model based on PCA do better?
> PCA based model accuracy generally lower than all variables or top 5 variables. Why?
> > ## Solution
> >
> > reducing variables in classifier, introduce bias error.
> {:.solution}
{:.challenge}


> ## Fit a PCA model with 5 PCs.
> What do you think the out come will be?
> > ## Solution
> >
> > upping variables in classifier, reduce bias error.
> > tail ends of distributions can have high predictive power - a small amount of variance can be impactful
> {:.solution}
{:.challenge}


# What Is Going On?

## Intuition:

#### PCA is a way to rotate the *axes* of your dataset around the *data* so that the axes line up with the *directions of the greatest variation* through the data.






# reviewed

1. exlpored Ames housing dataset
2. looked for variables that would correlate with/be good predictors for housing prices
3. indicated that PCA might be a way to approach this problem


We'll go into more detail on PCA in the next episode


```python

```