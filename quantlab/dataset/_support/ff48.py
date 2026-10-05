"""The Fama-French 48 industry classification of SIC codes.

Kenneth French publishes the 48 industries as lists of four-digit SIC ranges
(``Siccodes48.txt`` in ``Siccodes48.zip`` from his data library, the file
dated 2020-01-08). ``FF48_INDUSTRIES`` copies those ranges verbatim; no two
industries share a code. A SIC code inside no range is unclassified in
French's own data; here it goes to industry 48, "Other" (``OTHER``), so every
known SIC has an industry. An unknown SIC (NaN) stays NaN.

Thin industries can be merged into others by a *merge mapping*, pairs
``(from_code, to_code)`` applied after the classification
(``apply_merge``). ``check_merge`` refuses a code outside 1..48, an industry
merged into itself or into two targets, and a target that is itself merged
away, so one lookup always gives the final code.

Examples
--------
>>> ff48_codes(np.array([3571.0, 6021.0, 9995.0, np.nan])).tolist()
[35.0, 44.0, 48.0, nan]
>>> apply_merge(np.array([27.0, 28.0]), ((27, 28),)).tolist()
[28.0, 28.0]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The industry a SIC code inside no range is classified as: 48, "Other".
OTHER = 48

#: The largest four-digit SIC code.
_MAX_SIC = 9999


@dataclass(frozen=True)
class Ff48Industry:
    """One Fama-French 48 industry.

    Attributes
    ----------
    code : int
        The industry number, 1..48.
    short_name : str
        French's short name (``"Banks"``).
    name : str
        French's long name (``"Banking"``).
    sic_ranges : tuple of (int, int)
        The SIC ranges, both ends inclusive.
    """

    code: int
    short_name: str
    name: str
    sic_ranges: tuple[tuple[int, int], ...]


#: The 48 industries and their SIC ranges, verbatim from ``Siccodes48.txt``.
FF48_INDUSTRIES: tuple[Ff48Industry, ...] = (
    Ff48Industry(
        1,
        "Agric",
        "Agriculture",
        (
            (100, 199), (200, 299), (700, 799), (910, 919), (2048, 2048),
        ),
    ),
    Ff48Industry(
        2,
        "Food",
        "Food Products",
        (
            (2000, 2009), (2010, 2019), (2020, 2029), (2030, 2039), (2040, 2046),
            (2050, 2059), (2060, 2063), (2070, 2079), (2090, 2092), (2095, 2095),
            (2098, 2099),
        ),
    ),
    Ff48Industry(
        3,
        "Soda",
        "Candy & Soda",
        (
            (2064, 2068), (2086, 2086), (2087, 2087), (2096, 2096), (2097, 2097),
        ),
    ),
    Ff48Industry(
        4,
        "Beer",
        "Beer & Liquor",
        (
            (2080, 2080), (2082, 2082), (2083, 2083), (2084, 2084), (2085, 2085),
        ),
    ),
    Ff48Industry(
        5,
        "Smoke",
        "Tobacco Products",
        (
            (2100, 2199),
        ),
    ),
    Ff48Industry(
        6,
        "Toys",
        "Recreation",
        (
            (920, 999), (3650, 3651), (3652, 3652), (3732, 3732), (3930, 3931),
            (3940, 3949),
        ),
    ),
    Ff48Industry(
        7,
        "Fun",
        "Entertainment",
        (
            (7800, 7829), (7830, 7833), (7840, 7841), (7900, 7900), (7910, 7911),
            (7920, 7929), (7930, 7933), (7940, 7949), (7980, 7980), (7990, 7999),
        ),
    ),
    Ff48Industry(
        8,
        "Books",
        "Printing and Publishing",
        (
            (2700, 2709), (2710, 2719), (2720, 2729), (2730, 2739), (2740, 2749),
            (2770, 2771), (2780, 2789), (2790, 2799),
        ),
    ),
    Ff48Industry(
        9,
        "Hshld",
        "Consumer Goods",
        (
            (2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2843),
            (2844, 2844), (3160, 3161), (3170, 3171), (3172, 3172), (3190, 3199),
            (3229, 3229), (3260, 3260), (3262, 3263), (3269, 3269), (3230, 3231),
            (3630, 3639), (3750, 3751), (3800, 3800), (3860, 3861), (3870, 3873),
            (3910, 3911), (3914, 3914), (3915, 3915), (3960, 3962), (3991, 3991),
            (3995, 3995),
        ),
    ),
    Ff48Industry(
        10,
        "Clths",
        "Apparel",
        (
            (2300, 2390), (3020, 3021), (3100, 3111), (3130, 3131), (3140, 3149),
            (3150, 3151), (3963, 3965),
        ),
    ),
    Ff48Industry(
        11,
        "Hlth",
        "Healthcare",
        (
            (8000, 8099),
        ),
    ),
    Ff48Industry(
        12,
        "MedEq",
        "Medical Equipment",
        (
            (3693, 3693), (3840, 3849), (3850, 3851),
        ),
    ),
    Ff48Industry(
        13,
        "Drugs",
        "Pharmaceutical Products",
        (
            (2830, 2830), (2831, 2831), (2833, 2833), (2834, 2834), (2835, 2835),
            (2836, 2836),
        ),
    ),
    Ff48Industry(
        14,
        "Chems",
        "Chemicals",
        (
            (2800, 2809), (2810, 2819), (2820, 2829), (2850, 2859), (2860, 2869),
            (2870, 2879), (2890, 2899),
        ),
    ),
    Ff48Industry(
        15,
        "Rubbr",
        "Rubber and Plastic Products",
        (
            (3031, 3031), (3041, 3041), (3050, 3053), (3060, 3069), (3070, 3079),
            (3080, 3089), (3090, 3099),
        ),
    ),
    Ff48Industry(
        16,
        "Txtls",
        "Textiles",
        (
            (2200, 2269), (2270, 2279), (2280, 2284), (2290, 2295), (2297, 2297),
            (2298, 2298), (2299, 2299), (2393, 2395), (2397, 2399),
        ),
    ),
    Ff48Industry(
        17,
        "BldMt",
        "Construction Materials",
        (
            (800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661),
            (2950, 2952), (3200, 3200), (3210, 3211), (3240, 3241), (3250, 3259),
            (3261, 3261), (3264, 3264), (3270, 3275), (3280, 3281), (3290, 3293),
            (3295, 3299), (3420, 3429), (3430, 3433), (3440, 3441), (3442, 3442),
            (3446, 3446), (3448, 3448), (3449, 3449), (3450, 3451), (3452, 3452),
            (3490, 3499), (3996, 3996),
        ),
    ),
    Ff48Industry(
        18,
        "Cnstr",
        "Construction",
        (
            (1500, 1511), (1520, 1529), (1530, 1539), (1540, 1549), (1600, 1699),
            (1700, 1799),
        ),
    ),
    Ff48Industry(
        19,
        "Steel",
        "Steel Works Etc",
        (
            (3300, 3300), (3310, 3317), (3320, 3325), (3330, 3339), (3340, 3341),
            (3350, 3357), (3360, 3369), (3370, 3379), (3390, 3399),
        ),
    ),
    Ff48Industry(
        20,
        "FabPr",
        "Fabricated Products",
        (
            (3400, 3400), (3443, 3443), (3444, 3444), (3460, 3469), (3470, 3479),
        ),
    ),
    Ff48Industry(
        21,
        "Mach",
        "Machinery",
        (
            (3510, 3519), (3520, 3529), (3530, 3530), (3531, 3531), (3532, 3532),
            (3533, 3533), (3534, 3534), (3535, 3535), (3536, 3536), (3538, 3538),
            (3540, 3549), (3550, 3559), (3560, 3569), (3580, 3580), (3581, 3581),
            (3582, 3582), (3585, 3585), (3586, 3586), (3589, 3589), (3590, 3599),
        ),
    ),
    Ff48Industry(
        22,
        "ElcEq",
        "Electrical Equipment",
        (
            (3600, 3600), (3610, 3613), (3620, 3621), (3623, 3629), (3640, 3644),
            (3645, 3645), (3646, 3646), (3648, 3649), (3660, 3660), (3690, 3690),
            (3691, 3692), (3699, 3699),
        ),
    ),
    Ff48Industry(
        23,
        "Autos",
        "Automobiles and Trucks",
        (
            (2296, 2296), (2396, 2396), (3010, 3011), (3537, 3537), (3647, 3647),
            (3694, 3694), (3700, 3700), (3710, 3710), (3711, 3711), (3713, 3713),
            (3714, 3714), (3715, 3715), (3716, 3716), (3792, 3792), (3790, 3791),
            (3799, 3799),
        ),
    ),
    Ff48Industry(
        24,
        "Aero",
        "Aircraft",
        (
            (3720, 3720), (3721, 3721), (3723, 3724), (3725, 3725), (3728, 3729),
        ),
    ),
    Ff48Industry(
        25,
        "Ships",
        "Shipbuilding, Railroad Equipment",
        (
            (3730, 3731), (3740, 3743),
        ),
    ),
    Ff48Industry(
        26,
        "Guns",
        "Defense",
        (
            (3760, 3769), (3795, 3795), (3480, 3489),
        ),
    ),
    Ff48Industry(
        27,
        "Gold",
        "Precious Metals",
        (
            (1040, 1049),
        ),
    ),
    Ff48Industry(
        28,
        "Mines",
        "Non-Metallic and Industrial Metal Mining",
        (
            (1000, 1009), (1010, 1019), (1020, 1029), (1030, 1039), (1050, 1059),
            (1060, 1069), (1070, 1079), (1080, 1089), (1090, 1099), (1100, 1119),
            (1400, 1499),
        ),
    ),
    Ff48Industry(
        29,
        "Coal",
        "Coal",
        (
            (1200, 1299),
        ),
    ),
    Ff48Industry(
        30,
        "Oil",
        "Petroleum and Natural Gas",
        (
            (1300, 1300), (1310, 1319), (1320, 1329), (1330, 1339), (1370, 1379),
            (1380, 1380), (1381, 1381), (1382, 1382), (1389, 1389), (2900, 2912),
            (2990, 2999),
        ),
    ),
    Ff48Industry(
        31,
        "Util",
        "Utilities",
        (
            (4900, 4900), (4910, 4911), (4920, 4922), (4923, 4923), (4924, 4925),
            (4930, 4931), (4932, 4932), (4939, 4939), (4940, 4942),
        ),
    ),
    Ff48Industry(
        32,
        "Telcm",
        "Communication",
        (
            (4800, 4800), (4810, 4813), (4820, 4822), (4830, 4839), (4840, 4841),
            (4880, 4889), (4890, 4890), (4891, 4891), (4892, 4892), (4899, 4899),
        ),
    ),
    Ff48Industry(
        33,
        "PerSv",
        "Personal Services",
        (
            (7020, 7021), (7030, 7033), (7200, 7200), (7210, 7212), (7214, 7214),
            (7215, 7216), (7217, 7217), (7219, 7219), (7220, 7221), (7230, 7231),
            (7240, 7241), (7250, 7251), (7260, 7269), (7270, 7290), (7291, 7291),
            (7292, 7299), (7395, 7395), (7500, 7500), (7520, 7529), (7530, 7539),
            (7540, 7549), (7600, 7600), (7620, 7620), (7622, 7622), (7623, 7623),
            (7629, 7629), (7630, 7631), (7640, 7641), (7690, 7699), (8100, 8199),
            (8200, 8299), (8300, 8399), (8400, 8499), (8600, 8699), (8800, 8899),
            (7510, 7515),
        ),
    ),
    Ff48Industry(
        34,
        "BusSv",
        "Business Services",
        (
            (2750, 2759), (3993, 3993), (7218, 7218), (7300, 7300), (7310, 7319),
            (7320, 7329), (7330, 7339), (7340, 7342), (7349, 7349), (7350, 7351),
            (7352, 7352), (7353, 7353), (7359, 7359), (7360, 7369), (7370, 7372),
            (7374, 7374), (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378),
            (7379, 7379), (7380, 7380), (7381, 7382), (7383, 7383), (7384, 7384),
            (7385, 7385), (7389, 7390), (7391, 7391), (7392, 7392), (7393, 7393),
            (7394, 7394), (7396, 7396), (7397, 7397), (7399, 7399), (7519, 7519),
            (8700, 8700), (8710, 8713), (8720, 8721), (8730, 8734), (8740, 8748),
            (8900, 8910), (8911, 8911), (8920, 8999), (4220, 4229),
        ),
    ),
    Ff48Industry(
        35,
        "Comps",
        "Computers",
        (
            (3570, 3579), (3680, 3680), (3681, 3681), (3682, 3682), (3683, 3683),
            (3684, 3684), (3685, 3685), (3686, 3686), (3687, 3687), (3688, 3688),
            (3689, 3689), (3695, 3695), (7373, 7373),
        ),
    ),
    Ff48Industry(
        36,
        "Chips",
        "Electronic Equipment",
        (
            (3622, 3622), (3661, 3661), (3662, 3662), (3663, 3663), (3664, 3664),
            (3665, 3665), (3666, 3666), (3669, 3669), (3670, 3679), (3810, 3810),
            (3812, 3812),
        ),
    ),
    Ff48Industry(
        37,
        "LabEq",
        "Measuring and Control Equipment",
        (
            (3811, 3811), (3820, 3820), (3821, 3821), (3822, 3822), (3823, 3823),
            (3824, 3824), (3825, 3825), (3826, 3826), (3827, 3827), (3829, 3829),
            (3830, 3839),
        ),
    ),
    Ff48Industry(
        38,
        "Paper",
        "Business Supplies",
        (
            (2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955),
        ),
    ),
    Ff48Industry(
        39,
        "Boxes",
        "Shipping Containers",
        (
            (2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412),
        ),
    ),
    Ff48Industry(
        40,
        "Trans",
        "Transportation",
        (
            (4000, 4013), (4040, 4049), (4100, 4100), (4110, 4119), (4120, 4121),
            (4130, 4131), (4140, 4142), (4150, 4151), (4170, 4173), (4190, 4199),
            (4200, 4200), (4210, 4219), (4230, 4231), (4240, 4249), (4400, 4499),
            (4500, 4599), (4600, 4699), (4700, 4700), (4710, 4712), (4720, 4729),
            (4730, 4739), (4740, 4749), (4780, 4780), (4782, 4782), (4783, 4783),
            (4784, 4784), (4785, 4785), (4789, 4789),
        ),
    ),
    Ff48Industry(
        41,
        "Whlsl",
        "Wholesale",
        (
            (5000, 5000), (5010, 5015), (5020, 5023), (5030, 5039), (5040, 5042),
            (5043, 5043), (5044, 5044), (5045, 5045), (5046, 5046), (5047, 5047),
            (5048, 5048), (5049, 5049), (5050, 5059), (5060, 5060), (5063, 5063),
            (5064, 5064), (5065, 5065), (5070, 5078), (5080, 5080), (5081, 5081),
            (5082, 5082), (5083, 5083), (5084, 5084), (5085, 5085), (5086, 5087),
            (5088, 5088), (5090, 5090), (5091, 5092), (5093, 5093), (5094, 5094),
            (5099, 5099), (5100, 5100), (5110, 5113), (5120, 5122), (5130, 5139),
            (5140, 5149), (5150, 5159), (5160, 5169), (5170, 5172), (5180, 5182),
            (5190, 5199),
        ),
    ),
    Ff48Industry(
        42,
        "Rtail",
        "Retail",
        (
            (5200, 5200), (5210, 5219), (5220, 5229), (5230, 5231), (5250, 5251),
            (5260, 5261), (5270, 5271), (5300, 5300), (5310, 5311), (5320, 5320),
            (5330, 5331), (5334, 5334), (5340, 5349), (5390, 5399), (5400, 5400),
            (5410, 5411), (5412, 5412), (5420, 5429), (5430, 5439), (5440, 5449),
            (5450, 5459), (5460, 5469), (5490, 5499), (5500, 5500), (5510, 5529),
            (5530, 5539), (5540, 5549), (5550, 5559), (5560, 5569), (5570, 5579),
            (5590, 5599), (5600, 5699), (5700, 5700), (5710, 5719), (5720, 5722),
            (5730, 5733), (5734, 5734), (5735, 5735), (5736, 5736), (5750, 5799),
            (5900, 5900), (5910, 5912), (5920, 5929), (5930, 5932), (5940, 5940),
            (5941, 5941), (5942, 5942), (5943, 5943), (5944, 5944), (5945, 5945),
            (5946, 5946), (5947, 5947), (5948, 5948), (5949, 5949), (5950, 5959),
            (5960, 5969), (5970, 5979), (5980, 5989), (5990, 5990), (5992, 5992),
            (5993, 5993), (5994, 5994), (5995, 5995), (5999, 5999),
        ),
    ),
    Ff48Industry(
        43,
        "Meals",
        "Restaurants, Hotels, Motels",
        (
            (5800, 5819), (5820, 5829), (5890, 5899), (7000, 7000), (7010, 7019),
            (7040, 7049), (7213, 7213),
        ),
    ),
    Ff48Industry(
        44,
        "Banks",
        "Banking",
        (
            (6000, 6000), (6010, 6019), (6020, 6020), (6021, 6021), (6022, 6022),
            (6023, 6024), (6025, 6025), (6026, 6026), (6027, 6027), (6028, 6029),
            (6030, 6036), (6040, 6059), (6060, 6062), (6080, 6082), (6090, 6099),
            (6100, 6100), (6110, 6111), (6112, 6113), (6120, 6129), (6130, 6139),
            (6140, 6149), (6150, 6159), (6160, 6169), (6170, 6179), (6190, 6199),
        ),
    ),
    Ff48Industry(
        45,
        "Insur",
        "Insurance",
        (
            (6300, 6300), (6310, 6319), (6320, 6329), (6330, 6331), (6350, 6351),
            (6360, 6361), (6370, 6379), (6390, 6399), (6400, 6411),
        ),
    ),
    Ff48Industry(
        46,
        "RlEst",
        "Real Estate",
        (
            (6500, 6500), (6510, 6510), (6512, 6512), (6513, 6513), (6514, 6514),
            (6515, 6515), (6517, 6519), (6520, 6529), (6530, 6531), (6532, 6532),
            (6540, 6541), (6550, 6553), (6590, 6599), (6610, 6611),
        ),
    ),
    Ff48Industry(
        47,
        "Fin",
        "Trading",
        (
            (6200, 6299), (6700, 6700), (6710, 6719), (6720, 6722), (6723, 6723),
            (6724, 6724), (6725, 6725), (6726, 6726), (6730, 6733), (6740, 6779),
            (6790, 6791), (6792, 6792), (6793, 6793), (6794, 6794), (6795, 6795),
            (6798, 6798), (6799, 6799),
        ),
    ),
    Ff48Industry(
        48,
        "Other",
        "Almost Nothing",
        (
            (4950, 4959), (4960, 4961), (4970, 4971), (4990, 4991),
        ),
    ),
)

#: Every industry code, 1..48.
FF48_CODES: tuple[int, ...] = tuple(industry.code for industry in FF48_INDUSTRIES)


def _lookup() -> np.ndarray:
    """Return each SIC code's industry, indexed by the code (``OTHER`` outside every range)."""
    table = np.full(_MAX_SIC + 1, float(OTHER))
    for industry in FF48_INDUSTRIES:
        for low, high in industry.sic_ranges:
            table[low : high + 1] = industry.code
    return table


_LOOKUP = _lookup()


def ff48_codes(sic: np.ndarray) -> np.ndarray:
    """Return the Fama-French 48 industry of each SIC code.

    Parameters
    ----------
    sic : np.ndarray
        SIC codes as floats; NaN is an unknown code.

    Returns
    -------
    np.ndarray
        Float industry codes, 1..48, the same shape as ``sic``; ``OTHER``
        for a code inside no range (or outside 0..9999), NaN where ``sic``
        is NaN.

    Examples
    --------
    >>> ff48_codes(np.array([100.0, 4955.0, 1.0])).tolist()
    [1.0, 48.0, 48.0]
    """
    sic = np.asarray(sic, dtype=float)
    known = np.isfinite(sic)
    inside = known & (sic >= 0) & (sic <= _MAX_SIC)
    out = np.where(known, float(OTHER), np.nan)
    out[inside] = _LOOKUP[sic[inside].astype(int)]
    return out


def check_merge(merge, owner: str) -> tuple[tuple[int, int], ...]:
    """Check a merge mapping and return it as sorted ``(from_code, to_code)`` int pairs.

    Parameters
    ----------
    merge : iterable of (int, int)
        The pairs, each merging industry ``from_code`` into ``to_code``.
    owner : str
        Named in error messages.

    Raises
    ------
    ValueError
        If a pair is not two codes, a code is not an industry (1..48), an
        industry is merged into itself or into two targets, or a target is
        itself merged away.

    Examples
    --------
    >>> check_merge([[28, 27]], "demo")
    ((28, 27),)
    >>> check_merge([(27, 49)], "demo")
    Traceback (most recent call last):
    ...
    ValueError: demo: the industry merge mapping names unknown code(s) [49]; Fama-French 48 codes are 1..48.
    """
    pairs = []
    for pair in merge:
        pair = tuple(pair)
        if len(pair) != 2:
            raise ValueError(
                f"{owner}: each industry merge is a (from_code, to_code) pair, got {pair!r}."
            )
        pairs.append(pair)
    unknown = sorted(
        {
            code
            for pair in pairs
            for code in pair
            if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or code not in FF48_CODES
        },
        key=str,
    )
    if unknown:
        raise ValueError(
            f"{owner}: the industry merge mapping names unknown code(s) "
            f"{unknown}; Fama-French 48 codes are 1..48."
        )
    pairs = [(int(source), int(target)) for source, target in pairs]
    sources = [source for source, _ in pairs]
    if len(set(sources)) != len(sources):
        twice = sorted({source for source in sources if sources.count(source) > 1})
        raise ValueError(f"{owner}: industry/industries {twice} are merged into two targets.")
    itself = sorted(source for source, target in pairs if source == target)
    if itself:
        raise ValueError(f"{owner}: industry/industries {itself} are merged into themselves.")
    chained = sorted({target for _, target in pairs if target in set(sources)})
    if chained:
        raise ValueError(
            f"{owner}: merge target(s) {chained} are themselves merged away; "
            f"name the final industry instead."
        )
    return tuple(sorted(pairs))


def apply_merge(codes: np.ndarray, merge: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return ``codes`` with each merged industry replaced by its target.

    Parameters
    ----------
    codes : np.ndarray
        Float industry codes; NaN stays NaN.
    merge : tuple of (int, int)
        A mapping ``check_merge`` accepted.

    Examples
    --------
    >>> apply_merge(np.array([1.0, 2.0, np.nan]), ((1, 2),)).tolist()
    [2.0, 2.0, nan]
    """
    out = np.array(codes, dtype=float, copy=True)
    for source, target in merge:
        out[codes == source] = target
    return out


def short_names(merge: tuple[tuple[int, int], ...]) -> dict[str, str]:
    """Return the short name of each industry left after ``merge``, keyed by its code as text.

    Examples
    --------
    >>> names = short_names(((27, 28),))
    >>> len(names), names["28"], "27" in names
    (47, 'Mines', False)
    """
    merged_away = {source for source, _ in merge}
    return {
        str(industry.code): industry.short_name
        for industry in FF48_INDUSTRIES
        if industry.code not in merged_away
    }
