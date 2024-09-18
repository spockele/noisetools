"""Functions for interaction with HAWC2. Mostly focussed on Noise.

"""

import pandas as pd
import numpy as np
import os


def num_format(value: float, ljust: int = 9) -> str:
    """
    Function to convert numbers to the right format for the HAWC2 BL input file.

    Parameters
    ----------
    value: float
        Number to convert.
    ljust: int
        Number of total digits (including the decimal separator) before the OoM indicator.

    Returns
    -------
    String with the formatted number.
    """
    # Separate the sign of the number.
    sign = ' ' if value >= 0 else '-'
    # Make the smallest absolute value 1e-7, since HAWC2 doesn't like zeroes.
    value = max(abs(value), 1e-7)

    # Determine the OoM exponent.
    exp = 0 if value == 0. else np.log10(value)
    # Ensure that the exponent is always a rounded down integer.
    exp = int(exp) if exp >= 0 else int(np.floor(exp))

    # Separate the sign of the exponent.
    exp_sign = '+' if exp >= 0 else '-'
    # Determine the actual digits before the OoM indicator.
    num = str(round(value / (10 ** exp), ljust-2))

    # Compose the string to the form <sign><num>E<exp_sign><exp>
    return f'{sign}{num.ljust(ljust, "0")}E{exp_sign}{str(abs(exp)).zfill(2)}'


def read_hawc2_bldata(fpath: str | os.PathLike) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Read a HAWC2 boundary layer parameter file for Trailing edge noise.

    Parameters
    ----------
    fpath: str | os.PathLike
        Path to the HAWC2 BL data file to read.

    Returns
    -------
    Boolean indicating the type of BL data.
        - False: XFoil data
        - True: CFD data
    A pandas DataFrame with the boundary layer data from the bldata file.
        Index is a MultiIndex with columns: ['t/c (%)', 'Re (-)', 'AoA (deg)', 's/p'].
        Columns depends on the type of data that was used for the bldata file:
            - XFoil data (1): Index with indices: ['ue', 'cf', 'dpdx', 'delta', 'dstar', 'theta', 'xtr', 'xsep']
            - CFD data (2): MultiIndex with index: ['Ny', 'BL par']
                - Ny are the blade point indices, where 0 is used for ['ue', 'cf', 'dpdx', 'delta', 'dstar', 'theta',
                    'Xtr', 'xsep'], and 1 through Ny are used for the CFD data ['y', 'u', 'kt', 'eps'].
                - BL par are the names of the boundary layer parameters. This will be ['ue', 'cf', 'dpdx', 'delta',
                    'dstar', 'theta', 'Xtr', 'xsep'], followed by Ny repetitions of ['y', 'u', 'kt', 'eps'].

        See the HAWC2 manual (Sec. 12.15) [1]_ for more information about the BL input data file, the structure of
        which is roughly followed for the output DataFrame.

    The second pandas DataFrame contains the data of the airfoil thickness at 1, and 10 % of the chord (which is
    contained in the thickness lines of the BL data file).

    References
    ----------
    .. [1] T. J. Larsen and A. M. Hansen, `How 2 HAWC2, the user's manual', DTU, Department of Wind Energy, Roskilde,
        Denmark, Technical Report Riso-R-1597(ver. 13.0)(EN), May 2023. Available online:
        http://tools.windenergy.dtu.dk/HAWC2/manual/

    """
    # Open the file as lines.
    with open(fpath, 'r') as f:
        lines = f.readlines()

    # Set the base column names.
    cols = ['ue', 'cf', 'dpdx', 'delta', 'dstar', 'theta', 'xtr', 'xsep']
    # Extract information about CDF or XF and the number of points along the airfoil.
    cfd, npts = [int(val) for val in lines[4].split()]
    cfd = bool(cfd - 1)
    # Set the initial lines to skip
    skiplines = list(range(7))

    # In case of CFD data, the columns get a MultiIndex.
    if cfd:
        # Add the extra CFD columns.
        cols += ['y', 'u', 'kt', 'eps'] * npts
        # Generate the extra index layer for the points along the airfoil.
        pts = np.concat((np.repeat(0, 8),
                         np.repeat([n + 1 for n in range(npts)], 4)))

        # Create the MultiIndex.
        cols = pd.MultiIndex.from_arrays([pts, cols], names=['Ny', 'BL par'])

    # Extract info about number of thicknesses.
    n_thick = int(lines[6].split()[0])
    # Initialise lists for the index axis MultiIndex (thickness, reynolds number, angle of attack, top/bottom).
    th = []
    re = []
    aa = []
    tb = []

    t_1_10_index = []
    t_1_10 = []
    # Set the start index of the first Thickness.
    thickness_start: int = 7
    # Loop over the thicknesses.
    for tn in range(n_thick):
        # Extract the thickness.
        thickness = float(lines[thickness_start + 1].split()[0])
        t_1_10_index.append(thickness)
        t_1_10.append([float(val) for val in lines[thickness_start + 1][48:].split()])
        # Add the first lines of this thickness to be skipped later.
        skiplines += [thickness_start + n for n in range(4)]
        # Extract number of Reynolds numbers.
        n_reynolds = int(lines[thickness_start + 3])

        # Set the start index of the first Re.
        reynolds_start: int = thickness_start + 4
        # Loop over the Reynolds numbers
        for rn in range(n_reynolds):
            # Extract the Reynolds number.
            reynolds = float(lines[reynolds_start + 1].split()[0])
            # Extract the number of AoA's
            n_aoa = int(lines[reynolds_start + 3])
            # Add the first lines of this Re to be skipped later.
            skiplines += [reynolds_start + n for n in range(4)]
            # Add each AoA index line to be skipped later.
            skiplines += [reynolds_start + 4 + 3 * n for n in range(n_aoa)]

            # Set the start and stop index for this reynolds number.
            a_start, a_stop = reynolds_start + 4, reynolds_start + 4 + 3 * n_aoa

            # Add correct amount of thickness, Re, and top/bottom indices to the index lists.
            th += 2 * n_aoa * [thickness, ]
            re += 2 * n_aoa * [reynolds, ]
            tb += n_aoa * ['suction', 'pressure', ]
            # Add the AoA's to the index list.
            aa = np.append(aa, np.repeat([float(a.split()[0]) for a in lines[a_start:a_stop:3]], 2))

            # Set the start index for the next Reynolds number.
            reynolds_start += 3 * n_aoa + 4

        # Set the start index for the next thickness.
        thickness_start = reynolds_start

    # Remove the read lines.
    del lines

    t_1_10 = pd.DataFrame(t_1_10, dtype=float,
                          index=pd.Index(t_1_10_index, name='t/c (%)'),
                          columns=pd.Index([1, 10], name='x/c (%)'))

    # Read the file again, but parse the data. Uses the previously selected skiplines to only read the BL data.
    # This is more efficient than adding this info line-per-line.
    df = pd.read_csv(fpath, delimiter='\s+', skiprows=skiplines, header=None, dtype=float)
    # Set the column Index.
    df.columns = cols
    # Set the index MultiIndex.
    df.index = pd.MultiIndex.from_arrays([th, re, aa, tb], names=['t/c (%)', 'Re (-)', 'AoA (deg)', 's/p'])

    return cfd, df, t_1_10


def write_hawc2_bldata(fpath: str | os.PathLike, cfd: bool,
                       bl_data: pd.DataFrame, t_1_10: pd.DataFrame) -> None:
    """
    Write a HAWC2 boundary layer parameter file for Trailing edge noise.

    Parameters
    ----------
    fpath: str | os.PathLike
        Path to the HAWC2 BL data file to read.
    cfd: bool
        Boolean indicating the type of BL data.
            - False: XFoil data
            - True: CFD data
    bl_data: pandas.DataFrame
        A pandas DataFrame with the boundary layer data for the bldata file.
            Index is a MultiIndex with columns: ['t/c (%)', 'Re (-)', 'AoA (deg)', 's/p'].
            Columns depends on the type of data for the bldata file:
                - XFoil data (1): Index with indices: ['ue', 'cf', 'dpdx', 'delta', 'dstar', 'theta', 'xtr', 'xsep']
                - CFD data (2): MultiIndex with index: ['Ny', 'BL par']
                    - Ny are the blade point indices, where 0 is used for ['ue', 'cf', 'dpdx', 'delta', 'dstar',
                        'theta', 'Xtr', 'xsep'], and 1 through Ny are used for the CFD data ['y', 'u', 'kt', 'eps'].
                    - BL par are the names of the boundary layer parameters. This will be ['ue', 'cf', 'dpdx', 'delta',
                        'dstar', 'theta', 'Xtr', 'xsep'], followed by Ny repetitions of ['y', 'u', 'kt', 'eps'].

            !!WARNING: THE COLUMNS HAVE TO BE IN THE CORRECT ORDER FOR THE BLDATA FILE!!
            See the HAWC2 manual (Sec. 12.15) [1]_ for more information about the BL input data file, the structure of
            which is roughly followed for the output DataFrame.
    t_1_10: pandas.DataFrame
        A pandas DataFrame containing the data of the airfoil thickness at 1, and 10 % of the chord (which is contained
        in the thickness lines of the BL data file).

    References
    ----------
    .. [1] T. J. Larsen and A. M. Hansen, `How 2 HAWC2, the user's manual', DTU, Department of Wind Energy, Roskilde,
        Denmark, Technical Report Riso-R-1597(ver. 13.0)(EN), May 2023. Available online:
        http://tools.windenergy.dtu.dk/HAWC2/manual/

    """
    # Determine the number of points over the airfoil based on the input information.
    ny = 1 if not cfd else bl_data.columns.get_level_values(0)[-1]
    # Set the addition of a few things in case of CFD data.
    l2_cfd = 'CFD: Ny times Y,U,K_t,Epsi.' if cfd else ''
    # Extract the list of thicknesses from the input DataFrame.
    thicknesses = bl_data.index.unique(0)
    # Start the list with the lines for the BL data file with some header info.
    h2_te_file = [f'# Input data file for aero_noise module in HAWC2.\n',
                  f'# Data: Uedge, Cf, dP/dX, Delta, D^star, Theta, X_tr, X_sep [All -] on suct./pres. sides. '
                  f'{l2_cfd}\n',
                  f'# At x/C= 0.95 - Xtr_suct=  0.10 - Xtr_pres=  0.10 - N_crit=  3.0\n',
                  f'# BL data type (1: Xfoil - 2:CFD), Nb. of points for BL data (Must be 1 for Xfoil)\n',
                  f' {int(cfd) + 1} {str(ny).rjust(5, " ")}\n',
                  f'# Number of thicknesses: \n'
                  f'  {str(thicknesses.size).rjust(2, " ")}\n'
                  ]
    # Loop over the thicknesses.
    for ti, thickness in enumerate(thicknesses):
        # Add the header for the current airfoil to the file.
        h2_te_file.append(f'# Thickness no. {ti + 1}\n')
        # Add the actual thickness ration in percent.
        # Note: the HAWC2 thickness correction requires the two thickness values in the comment of this line,
        #   after character nr. 48. Make sure there is nothing else past character nr. 48.
        h2_te_file.append(f' {num_format(thickness, ljust=6)}  # [% Chord] - At 1 and 10% Chord: '
                          f'{num_format(t_1_10.loc[thickness, 1], ljust=6)} '
                          f'{num_format(t_1_10.loc[thickness, 10], ljust=6)}\n')

        # Extract the list of Reynolds number for this thickness.
        reynolds_numbers = bl_data.loc[thickness, :].index.unique(0)
        # Add the number of Reynolds numbers for this airfoil to the BL input file.
        h2_te_file.append(f'# Number of Reynolds Numbers (Thickness no. {ti + 1})\n')
        h2_te_file.append(f'  {str(reynolds_numbers.size).rjust(2, " ")}\n')

        # Loop over the Reynolds numbers.
        for ri, re in enumerate(reynolds_numbers):
            # Get the list of angles of attack for this Reynolds number.
            angles_of_attack = bl_data.loc[thickness, :].loc[re, :].index.unique(0)
            # Add the header with Reynolds number and number of AoAs to the BL input file.
            h2_te_file.append(f'# Reynolds Number no. {ri + 1} (Thickness no. {ti + 1})\n')
            h2_te_file.append(f' {num_format(re, ljust=6)}  # [-]\n')
            h2_te_file.append(f'# Number of Angles of Attack (Reynolds Number no. {ri + 1}, Thickness no. {ti + 1})\n')
            h2_te_file.append(f'  {str(angles_of_attack.size).rjust(2, " ")}\n')

            # Loop over the angles of attack.
            for ai, aoa in enumerate(angles_of_attack):
                # Add line with converted AoA.
                h2_te_file.append(f' {num_format(aoa, ljust=6)}  # AoA {ai + 1} [deg]\n')
                # Create the string representation of the data for the suction side for this t, Re, AoA
                data_suct = bl_data.loc[thickness, :].loc[re, :].loc[aoa, :].loc["suction", :]
                data_suct = data_suct.to_string(header=False, index=False, float_format=num_format).split('\n')
                h2_te_file.append(f' {" ".join(data_suct)}\n')
                # Create the string representation of the data for the pressure side for this t, Re, AoA
                data_pres = bl_data.loc[thickness, :].loc[re, :].loc[aoa, :].loc["pressure", :]
                data_pres = data_pres.to_string(header=False, index=False, float_format=num_format).split('\n')
                h2_te_file.append(f' {" ".join(data_pres)}\n')

    # Write all these lines to the data file.
    with open(fpath, 'w', encoding='utf8') as f:
        f.writelines(h2_te_file)
