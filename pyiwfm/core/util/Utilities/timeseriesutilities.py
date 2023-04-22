"""
timeseries_utilities.py
Author: Tyler Hatch PhD, PE

This is the timeseries_utilities module of the python version of IWFM
"""
import numpy as np

from pyiwfm.core.util.Utilities.general_utilities import (
    count_occurence,
    first_location,
    upper_case,
)
from pyiwfm.core.util.Utilities.message_logger import (
    log_message,
    FATAL,
    set_last_message,
)

MODNAME = "TimeSeriesUtilities::"

TIMESTEP_LENGTH = 16
cache_limit = 500
simulation_timestep_inminutes = 0


class TimeStep:
    def __init__(
        self,
        track_time=False,
        current_time=0.0,
        end_time=0.0,
        current_timestep=0,
        current_date_and_time="",
        end_date_and_time="",
        deltat=0.0,
        deltat_inminutes=0,
        unit="",
    ):
        self.track_time = track_time
        self.current_time = current_time
        self.end_time = end_time
        self.current_timestep = current_timestep
        self.current_date_and_time = current_date_and_time
        self.end_date_and_time = end_date_and_time
        self.deltat = deltat
        self.deltat_inminutes = deltat_inminutes
        self.unit = unit


RECOGNIZED_INTERVALS_INMINUTES = [
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
    60,
    120,
    180,
    240,
    360,
    480,
    720,
    1440,
    10080,
    43200,
    525600,
]

RECOGNIZED_INTERVALS = [
    "1MIN",
    "2MIN",
    "3MIN",
    "4MIN",
    "5MIN",
    "10MIN",
    "15MIN",
    "20MIN",
    "30MIN",
    "1HOUR",
    "2HOUR",
    "3HOUR",
    "4HOUR",
    "6HOUR",
    "8HOUR",
    "12HOUR",
    "1DAY",
    "1WEEK",
    "1MON",
    "1YEAR",
]


def is_time_interval_valid(interval):
    """
    Return the index of the time interval if valid
    """
    if interval not in RECOGNIZED_INTERVALS:
        return None

    return RECOGNIZED_INTERVALS.index(interval)


def is_leap_year(year):
    """
    Check if year is a leap year
    """
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False


def julian_date_to_day_month_year(julian_date):
    """
    Convert julian date to day month year

    Parameters
    ----------
    julian_date : int
        julian date to convert

    Returns
    -------
    tuple
        day, month, year
    """
    l = int(julian_date + 68569)
    n = int(4 * l / 146097)
    l = l - int((146097 * n + 3) / 4)
    year = int(4000 * (l + 1) / 1461001)
    l = l - int(1461 * year / 4) + 31
    month = int(80 * l / 2447)
    day = l - int(2447 * month / 80)
    l = int(month / 11)
    month = month + 2 - 12 * l
    year = 100 * (n - 49) + year + l

    return day, month, year


def day_month_year_to_julian_date(day, month, year, return_status=False):
    """
    Convert day, month, year to julian day

    this approach can only convert a date after 1 Jan 0001 to julian day

    Parameters
    ----------
    day : int
        integer day of month

    month : int
        integer month of year

    year : int
        integer year

    return_status : bool default False
        flag to determine if status is also returned

    Returns
    -------
    int
        julian date
    """
    this_procedure = MODNAME + "DayMonthYearToJulianDate"

    # make sure year is not less than 1
    if year < 1:
        log_message(
            "Cannot convert dates with calendar year less than 1 to Julian day!",
            FATAL,
            this_procedure,
        )

    # make sure month is between 1 and 12
    if month < 1 or month > 12:
        log_message(f"Incorrect number for month {month}!", FATAL, this_procedure)

    # make sure day is correct
    if month in [1, 3, 5, 7, 8, 10, 12]:
        if day < 1 or day > 31:
            error_code = 1
        else:
            error_code = 0

    elif month in [4, 6, 9, 11]:
        if day < 1 or day > 30:
            error_code = 1
        else:
            error_code = 0

    else:
        if is_leap_year(year):
            if day < 1 or day > 29:
                error_code = 1
            else:
                error_code = 0

        else:
            if day < 1 or day > 28:
                error_code = 1
            else:
                error_code = 0

    if error_code != 0:
        log_message(
            f"Day {day} of the month is incorrect given the month {month}!",
            FATAL,
            this_procedure,
        )

    julian_date = (
        day
        - 32075
        + int(1461 * (year + 4800 + int((month - 14) / 12)) / 4)
        + int(367 * (month - 2 - int((month - 14) / 12) * 12) / 12)
        - int(3 * int((year + 4900 + int((month - 14) / 12)) / 100) / 4)
    )

    if not return_status:
        return julian_date

    return julian_date, error_code


def leap_year_correction(timestamp):
    """
    Correct timestamp for leap year

    Parameters
    ----------
    timestamp : str
        timestamp to check and correct for leap year

    Returns
    -------
    str
        corrected timestamp accounting for leap year
    """
    this_procedure = MODNAME + "LeapYearCorrection"

    # set output timestamp equal to timestamp
    out_timestamp = timestamp

    # check if timestamp refers to February
    if extract_month(timestamp) != 2:
        return timestamp

    # check if timestamp refer February 29
    if extract_day(timestamp) != 29:
        return timestamp

    # get the year
    year = extract_year(timestamp)

    # make sure that year of timestamp does not use the 4000 flag for the year
    if year == 4000:
        message_array = []
        message_array.append(f"Time stamp {timestamp} cannot be corrected for ")
        message_array.append("leap year because it includes the Year 4000 flag!")
        log_message(message_array, FATAL, this_procedure)

    # correct February 29 if it is a leap year
    if not is_leap_year(year):
        return timestamp.replace("/29/", "/28/")


def is_timestamp_valid(aline):
    """
    Check if a string is a valid timestamp

    Parameters
    ----------
    aline : str
        value to check if valid timestamp

    Returns
    -------
    bool
        True if string is valid timestamp else False
    """
    # strip timestamp from string
    timestamp = strip_timestamp(aline)

    # check if propective timestamp has 2 "/"
    if count_occurence("/", timestamp) != 2:
        return False

    # check if prospective timestamp has 1 "_"
    if count_occurence("_", timestamp) != 1:
        return False

    # check if prospective timestamp has 1 ":"
    if count_occurence(":") != 1:
        return False

    # check if length is equal to the timestep length
    if len(timestamp.strip()) != TIMESTEP_LENGTH:
        return False

    return True


def strip_timestamp(aline):
    """
    Strips a timestamp from a string

    Parameters
    ----------
    aline : str
        string containing timestamp to extract

    Returns
    -------
    str or None
        timestamp
    """
    # check if the line includes "/"
    location_of_first_slash = first_location("/", aline)

    #
    if not location_of_first_slash:
        return None

    # beginning of timestamp should be two characters before the first slash
    # otherwise the first character of the string
    timestamp_begin = max(location_of_first_slash - 2, 0)

    # end of timestamp should be the timestep length past the beginning of the
    # timestamp but no less than the beginning of the timestamp
    timestamp_end = max(timestamp_begin + TIMESTEP_LENGTH, timestamp_begin)

    # slice timestamp from string line
    timestamp = aline[timestamp_begin, timestamp_end]

    return timestamp


def extract_month(timestamp):
    """
    Extract month from timestamp string

    Parameters
    ----------
    timestamp : str
        timestamp to extract month

    Returns
    -------
    int
        integer month from timestamp
    """
    return int(timestamp[:2])


def extract_day(timestamp):
    """
    Extract day from timestamp string

    Parameters
    ----------
    timestamp : str
        timestamp to extract day

    Returns
    -------
    int
        integer day from timestamp
    """
    return int(timestamp[3:5])


def extract_year(timestamp):
    """
    Extract year from timestamp string

    Parameters
    ----------
    timestamp : str
        timestamp to extract year

    Returns
    -------
    int
        integer year from timestamp string
    """
    return int(timestamp[6:10])


def extract_hour(timestamp):
    """
    Extract hour from timestamp string

    Parameters
    ----------
    timestamp : str
        timestamp to extract hour

    Returns
    -------
    int
        integer hour from timestamp string
    """
    return int(timestamp[11:13])


def extract_minute(timestamp):
    """
    Extract minute from timestamp string

    Parameters
    ----------
    timestamp : str
        timestamp to extract minute

    Returns
    -------
    int
        integer minute from timestamp string
    """
    return int(timestamp[14:])


def timestamp_to_julian_date_and_minutes(timestamp, return_status=False):
    """
    Extract julian date and minutes past midnight from timestamp

    Parameters
    ----------
    timestamp : str
        timestamp to convert

    Returns
    -------
    tuple[int, int]
        if return_status = False, julian date and number of minutes past midnight

    tuple[int, int, int]
        if return_status = True, julian date, number of minutes past midnight, and status
    """
    this_procedure = MODNAME + "TimeStampToJulianDateAndMinutes"

    # convert date to julian date
    julian_date, error_code = day_month_year_to_julian_date(
        extract_day(timestamp),
        extract_month(timestamp),
        extract_year(timestamp),
        return_status=True,
    )

    # if return_status is True, then return the status code otherwise throw error
    # when error_code is not equal to 0 (success)
    if not return_status and error_code != 0:
        log_message(
            "Error in converting simulation date to Julian date",
            FATAL,
            this_procedure,
        )
        return

    status = error_code

    minutes_after_midnight = int(extract_hour(timestamp) * 60) + extract_minute(
        timestamp
    )

    if return_status:
        return julian_date, minutes_after_midnight, status

    return julian_date, minutes_after_midnight


def dssstyledate_from_daymonthyear(day, month, year):
    """
    Convert day, month, year to dss style date

    Parameters
    ----------
    day : int
        integer day

    month : int
        integer month

    year : int
        integer year

    Returns
    -------
    str
        dss style date
    """
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    return f"{day:02d}{months[month-1]}{year}"


def dssstyledate_from_timestamp(timestamp):
    """
    Convert timestamp to DSS-style date

    Parameters
    ----------
    timestamp : str
        timestamp to convert to DSS-style date

    Returns
    -------
    str
        dss style date
    """
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    day = extract_day(timestamp)
    month = extract_month(timestamp)
    year = extract_year(timestamp)

    return f"{day} {months[month-1]} {year}"


def dssstylehours_aftermidnight(timestamp):
    """
    Get DSS style hours after midnight from timestamp

    Parameters
    ----------
    timestamp : str
        timestamp to extract DSS-style hours after midnight

    Returns
    -------
    str
        DSS-style hours after midnight
    """
    hours = extract_hour(timestamp)
    minutes = extract_minute(timestamp)

    return f"{hours}{minutes}"


def julian_date_and_minutes_to_timestamp(julian_date, minutes_after_midnight):
    """
    Convert julian date and minutes after midnight to timestamp

    Parameters
    ----------
    julian_date : int
        julian date to convert to timestamp

    minutes_after_midnight : int
        number of minutes after midnight

    Return
    ------
    str
        timestamp
    """
    hours = int(minutes_after_midnight / 60)
    minutes = int(minutes_after_midnight - hours * 60)

    if hours == 0 and minutes == 0:
        hours = 24
        temp_julian_date = julian_date - 1
    else:
        temp_julian_date = julian_date

    day, month, year = julian_date_to_day_month_year(temp_julian_date)

    return f"{month:02d}/{day:02d}/{year}_{hours}:{minutes:02d}"


def timestamp_to_julian(timestamp, return_status=False):
    """
    Convert timestamp to fractional julian time

    Parameters
    ----------
    timestamp : str
        time stamp of date and time

    Returns
    -------
    float
        if return_status is False, julian date including fractional day

    tuple[float, int]
        if return_status is True, julian date including fractional day and status
    """
    julian_date, minutes_after_midnight, status = timestamp_to_julian_date_and_minutes(
        timestamp, return_status=True
    )

    julian = float(julian_date) + minutes_after_midnight / 1440

    if return_status:
        return julian, status

    return julian


def julian_to_timestamp(julian):
    """
    Convert fractional julian time to timestamp

    Parameters
    ----------
    julian : float
        fractional julian date and time

    Returns
    -------
    str
        timestamp
    """
    # get full julian day representation of julian date
    julian_date = int(julian)

    # get number of minutes after midnight
    minutes_after_midnight = int(1440 * (julian - julian_date))

    return julian_date_and_minutes_to_timestamp(julian_date, minutes_after_midnight)


def get_julian_dates_between_timestamps_with_time_increment(
    interval_inminutes, begin_date_and_time, end_date_and_time
):
    """
    Get a list of julian dates between two time stamps using a time increment

    Parameters
    ----------
    interval_inminutes : int
        time interval between time stamps used to generate list

    begin_date_and_time : str
        time stamp of beginning date and time

    end_date_and_time : str
        time stamp of ending date and time

    Returns
    -------
    np.ndarray
    """
    timestamp = begin_date_and_time
    julian_dates = []
    while check_for_less_than(timestamp, end_date_and_time):
        jd = timestamp_to_julian(timestamp)
        julian_dates.append(jd)
        timestamp = increment_timestamp(timestamp, interval_inminutes, 1)

    return np.array(julian_dates)


def adjust_timestamp_with_year_4000(adjusted_timestamp, timestamp):
    """
    Adjust the year 4000 flag with the year of another time stamp

    Parameters
    ----------
    adjusted_timestamp : str
        timestamp to be adjusted if 4000 used as placeholder for year

    timestamp : str
        timestamp to use when adjusting the timestamp from 4000 to actual year

    Returns
    -------
    tuple[str, bool]
        adjusted timestamp and True if 4000 flag used otherwise False
    """
    if extract_year(adjusted_timestamp) == 4000:
        # set year 4000 flag to True
        year4000flag = True

        # get year from timestamp
        year = extract_year(timestamp)

        # replace 4000 with year in adjusted_timestamp
        adjusted_timestamp.replace("4000", str(year))

        return adjusted_timestamp, year4000flag

    return adjusted_timestamp, False


def timestamp_to_year_4000(timestamp):
    """
    Convert timestamp to timestamp with year 4000 flag

    Parameters
    ----------
    timestamp : str
        timestamp to convert from actual date to date with 4000 flag as year

    Returns
    -------
    str
        converted timestamp with year 4000 flag
    """
    # get year to replace from timestamp
    year = extract_year(timestamp)

    return timestamp.replace(str(year), "4000")


def increment_timestamp(timestamp, interval_inminutes, num_intervals=1):
    """
    Increment a timestamp by a certain number of minutes

    Parameters
    ----------
    timestamp : str
        timestamp to increment

    interval_inminutes : int
        number of minutes in an interval of time to increment timestamp

    num_intervals : int default 1
        number of intervals to increment time

    Returns
    -------
    str
        timestamp incremented by number of intervals of the specified interval length
    """
    # convert timestamp to julian date and number of minutes after midnight
    julian_date, minutes_after_midnight = timestamp_to_julian_date_and_minutes(
        timestamp
    )

    # increment julian date and minutes past midnight
    sign = 1
    local_interval = interval_inminutes

    if interval_inminutes < 0:
        sign = -1
        local_interval = abs(interval_inminutes)

    (
        end_julian_date,
        end_minutes_after_midnight,
    ) = increment_julian_date_and_minutes_after_midnight(
        local_interval, sign * num_intervals, julian_date, minutes_after_midnight
    )

    return julian_date_and_minutes_to_timestamp(
        end_julian_date, end_minutes_after_midnight
    )


def increment_julian_date_and_minutes_after_midnight(
    interval_inminutes, num_intervals, begin_julian_date, begin_minutes_after_midnight
):
    """
    Increment a julian date and number of minutes after midnight by a certain number of minutes

    Parameters
    ----------
    interval_inminutes : int
        number of minutes making up an interval of time

    num_intervals : int
        number of time intervals to increment in time

    begin_julian_date : int
        julian date to increment by the specified number of intervals

    begin_minutes_after_midnight : int
        number of minutes after midnight to use when incrementing time

    Returns
    -------
    tuple[int, int]
        julian date and minutes after midnight after incrementing by a
        specified number of intervals of a certain size
    """
    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # interval is less than or equal to a week (60*24*7)
    if interval_inminutes <= 10080:

        # get number of minutes to increment
        all_minutes = interval_inminutes * num_intervals

        # get number of days
        n_days = int(all_minutes / 1440)

        # get number of minutes
        n_minutes = all_minutes - int(n_days * 1440)

        # increment values
        end_julian_date = begin_julian_date + n_days
        end_minutes_after_midnight = begin_minutes_after_midnight + n_minutes

        # check dates for adjustments i.e. minutes after midnight exceeding a day
        if end_minutes_after_midnight > 1440:
            end_julian_date += 1
            end_minutes_after_midnight -= 1440
        elif end_minutes_after_midnight < 1:
            end_julian_date -= 1
            end_minutes_after_midnight += 1440

    # interval is greater than or equal to smallest month i.e. 28 days (60*24*28)
    elif interval_inminutes >= 40320:
        # get day, month, and year
        day, month, year = julian_date_to_day_month_year(begin_julian_date)

        # determine if last day in month
        is_last_day = False
        if month in [1, 3, 5, 7, 8, 10, 12]:
            if day == 31:
                is_last_day = True

        elif month in [4, 6, 9, 11]:
            if day == 30:
                is_last_day = True

        else:
            if is_leap_year(year):
                if day == 29:
                    is_last_day = True

            else:
                if day == 28:
                    is_last_day = True

        # find number of years and months to increment
        if interval_inminutes >= 40320 and interval_inminutes <= 44640:
            n_years = int(num_intervals / 12)
            n_months = num_intervals - int(n_years * 12)

        else:
            n_years = num_intervals
            n_months = 0

        # increment date
        year += n_years
        month += n_months

        # check month is valid
        if month > 12:
            year += 1
            month -= 12
        elif month < 1:
            year -= 1
            month += 12

        # set day to end of month for monthly or annual intervals
        if is_last_day:
            if begin_minutes_after_midnight == 1440:
                if month == 2:
                    if is_leap_year(year):
                        day = 29
                    else:
                        day = 28
                else:
                    day = days_in_month[month - 1]

        # correct day if in february and greater than 28 for non-leap year or 29 for leap year
        if month == 2:
            if day > 28:
                if is_leap_year(year):
                    day = 29

                else:
                    day = 28

        # compute the ending julian date
        end_julian_date = day_month_year_to_julian_date(day, month, year)

        end_minutes_after_midnight = begin_minutes_after_midnight

    return end_julian_date, end_minutes_after_midnight


def set_simulation_timestep(deltat_inminutes):
    """
    Set simulation time step length

    Parameters
    ----------
    deltat_inminutes : int
        number of minutes corresponding to the simulation timestep

    Returns
    -------
    None
        sets module variable simulation_timestep_inminutes
    """
    global simulation_timestep_inminutes

    simulation_timestep_inminutes = deltat_inminutes


def set_cache_limit(cache):
    """
    Set the cache size for the time series data output

    Parameters
    ----------
    cache : int
        cache size for storing time series data

    Return
    ------
    None
        sets module variable simulation_timestep_inminutes
    """
    global cache_limit

    cache_limit = cache


def get_cache_limit():
    """
    get the cache size for time series data output

    Returns
    -------
    int
        cache size for time series data
    """
    return cache_limit


def set_tsd_cache_size(file_name, n_columns, n_rows=1):
    """
    Set in-memory storage of timeseries data before print-out
    
    Parameters
    ----------
    file_name : str
        output file name for time series data
    
    n_columns : int
        number of columns of timeseries data
        
    n_rows : int, default 1
        number of rows of timeseries data
    
    Returns
    -------
    """
    this_procedure = MODNAME + "SetTSDCacheSize"

    if n_rows * n_columns > cache_limit:
        n_batch = 1
    else:
        n_batch = cache_limit / (n_rows * n_columns)

    # Check if the storage arrays are already defined
    if 'values_for_output' in locals():
        del values_for_output
        if 'time_array' in locals():
            del time_array

    # Allocate memory for the data storage array
    values_for_output = np.empty((n_rows, n_columns, n_batch))

    # Allocate memory for the time storage array
    if 'time_array' in locals():
        time_array = np.empty(n_batch)

    # Set the data fields
    number_of_data_batch = n_batch
    number_of_data_rows = n_rows
    number_of_data_columns = n_columns

    return values_for_output, number_of_data_batch, number_of_data_rows, number_of_data_columns, time_array


def adjust_rate_type_data(r, rate_type_data_array, conversion_factor=None, data_interval=None, last_data_date=None):
    """
    Modify the time series rate type input data so that its time unit is consistent with simulation timestep
    
    Parameters
    ----------
    r : np.ndarray
        2D array of float containing time series rate type input data
        
    rate_type_data_array : np.ndarray
        1D array of bool containing 

    conversion_factor : float or None, default None
        conversion factor between time series rate type units and simulation units

    data_interval : int or None, default None
        time step length in minutes

    last_data_date : str or None, default None
        timestamp for last data read

    Returns
    -------
    np.ndarray
        2D array of float containing adjusted time series rate type data
    """
    # get the dimensions of the rate type data
    nrow, ncol = r.shape
    r1d = r.flatten()

    if conversion_factor is not None:
        factor = conversion_factor
    
    elif data_interval is not None and last_data_date is not None:
        if data_interval == 0:
            factor = 1.0
        else:
            # decrement the time stamp of the data last read by the data interval in minutes
            timestamp_begin = increment_timestamp(last_data_date, -data_interval)

            # compute conversion factor
            factor = float(n_periods(simulation_timestep_inminutes, timestamp_begin, last_data_date))

    # convert
    if rate_type_data_array.size == 1:
        # If RateTypeDataArray has only one element
        if rate_type_data_array[0]:
            # If the element is True, divide the entire input array r by the conversion factor
            r /= factor
    else:
        # If RateTypeDataArray has more than one element
        # Update the flattened array r1d by dividing only the selected elements by the conversion factor
        r1d[rate_type_data_array] /= factor
        
        # Reshape the flattened array r1d back to 2D array r with the original shape, using column-major order
        r = r1d.reshape((nrow, ncol), order='F')

    return r  # Return the adjusted array r

    





def time_units_check_less_than_or_equal(time_unit1, time_unit2):
    """
    Check if one time unit is less than or equal to another

    Parameters
    ----------
    time_unit1 : str
        time unit to compare

    time_unit2 : str
        time unit to compare against

    Returns
    -------
    bool
        True if time_unit1 is less than or equal to time_unit2 otherwise False
    """
    time_unit1_inminutes, _ = ctimestep_to_rtimestep(time_unit1)
    time_unit2_inminutes, _ = ctimestep_to_rtimestep(time_unit2)

    if time_unit1_inminutes <= time_unit2_inminutes:
        return True

    return False


def check_for_less_than(timestamp1, timestamp2):
    """
    Check if one timestamp is less than another

    Parameters
    ----------
    timestamp1 : str
        timestamp to check if less than another

    timestamp2 : str
        timestamp to compare to see if other is less

    Returns
    -------
    bool
        True if timestamp1 is less than timestamp2, otherwise False
    """
    julian_date1, minutes_after_midnight1 = timestamp_to_julian_date_and_minutes(
        timestamp1
    )
    julian_date2, minutes_after_midnight2 = timestamp_to_julian_date_and_minutes(
        timestamp2
    )

    if julian_date1 < julian_date2:
        return True
    elif julian_date1 == julian_date2:
        if minutes_after_midnight1 < minutes_after_midnight2:
            return True

    return False


def check_for_greater_than(timestamp1, timestamp2):
    """
    Check if one timestamp is greater than another

    Parameters
    ----------
    timestamp1 : str
        timestamp to check if greater than another

    timestamp2 : str
        timestamp to compare to see if other is greater

    Returns
    -------
    bool
        True if timestamp1 is greater than timestamp2, otherwise False
    """
    julian_date1, minutes_after_midnight1 = timestamp_to_julian_date_and_minutes(
        timestamp1
    )
    julian_date2, minutes_after_midnight2 = timestamp_to_julian_date_and_minutes(
        timestamp2
    )

    if julian_date1 > julian_date2:
        return True
    elif julian_date1 == julian_date2:
        if minutes_after_midnight1 > minutes_after_midnight2:
            return True

    return False


def check_for_greater_than_or_equal_to(timestamp1, timestamp2):
    """
    Check if one timestamp is greater than or equal to another

    Parameters
    ----------
    timestamp1 : str
        timestamp to check if greater than or equal to another

    timestamp2 : str
        timestamp to compare to see if other is greater than or equal

    Returns
    -------
    bool
        True if timestamp1 is greater than or equal to timestamp2, otherwise False
    """
    julian_date1, minutes_after_midnight1 = timestamp_to_julian_date_and_minutes(
        timestamp1
    )
    julian_date2, minutes_after_midnight2 = timestamp_to_julian_date_and_minutes(
        timestamp2
    )

    if julian_date1 >= julian_date2:
        return True
    elif julian_date1 == julian_date2:
        if minutes_after_midnight1 >= minutes_after_midnight2:
            return True

    return False


def n_periods_between_timestamps(deltat_inminutes, begin_timestamp, end_timestamp):
    """
    Compute the number of periods between two timestamps

    Parameters
    ----------
    deltat_inminutes : int
        number of minutes for the time increment between two timestamps

    begin_timestamp : str
        starting timestamp for determining number of periods

    end_timestamp : str
        ending timestamp for determining number of periods

    Returns
    -------
    int
        number of periods of length deltat_inminutes between begin_timestamp and end_timestamp
    """
    # convert to julian dates and minutes after midnight
    (
        begin_julian_date,
        begin_minutes_after_midnight,
    ) = timestamp_to_julian_date_and_minutes(begin_timestamp)
    end_julian_date, end_minutes_after_midnight = timestamp_to_julian_date_and_minutes(
        end_timestamp
    )

    # get day, month, and year if interval is greater than or equal to a month
    if deltat_inminutes >= 40320:
        begin_day, begin_month, begin_year = julian_date_to_day_month_year(
            begin_julian_date
        )
        end_day, end_month, end_year = julian_date_to_day_month_year(end_julian_date)

    # monthly timestamp
    if deltat_inminutes >= 40320 and deltat_inminutes <= 44640:
        n_period = (
            ((end_year - begin_year) * 12)
            + (end_month - begin_month)
            + int((end_day - begin_day) / 27)
        )
        temp_timestamp = increment_timestamp(
            begin_timestamp, deltat_inminutes, n_period
        )

        if check_for_greater_than(temp_timestamp, end_timestamp):
            n_period -= 1

    elif deltat_inminutes > 525600:
        n_period = (end_year - begin_year) + int(
            ((end_month - begin_month) + int((end_day - begin_day) / 28)) / 12
        )
        temp_timestamp = increment_timestamp(
            begin_timestamp, deltat_inminutes, n_period
        )

        if check_for_greater_than(temp_timestamp, end_timestamp):
            n_period -= 1

    else:
        n_period = int(
            (
                ((end_julian_date - begin_julian_date) * 1440)
                + (end_minutes_after_midnight - begin_minutes_after_midnight)
            )
            / deltat_inminutes
        )

    return n_period


def n_periods_between_times(delta_t, begin_time, end_time):
    """
    Compute number of periods between two times

    Parameters
    ----------
    delta_t : float
        timestep between begin and end times

    begin_time : float
        start time for determining number of periods

    end_time : float
        end time for determining number of periods

    Returns
    -------
    int
        number of periods of length delta_t between begin_time and end_time
    """
    return (end_time - begin_time) / delta_t

def n_periods(deltat, begin_time_or_timestamp, end_time_or_timestamp):
    """
    Interface for calling n_periods_between_times or n_periods_between_timestamps

    Parameters
    ----------
    deltat : int or float
        number of minutes between timestamps or timestep between begin and end times

    begin_time_or_timestamp : str or float
        beginning timestamp or start time

    end_time_or_timestamp : str or float
        ending timestamp or end time
    """
    if isinstance(deltat, int) and isinstance(begin_time_or_timestamp, str) and isinstance(end_time_or_timestamp, str):
        return n_periods_between_timestamps(deltat, begin_time_or_timestamp, end_time_or_timestamp)

    elif isinstance(deltat, float) and isinstance(begin_time_or_timestamp, float) and isinstance(end_time_or_timestamp, float):
        return n_periods_between_times(deltat, begin_time_or_timestamp, end_time_or_timestamp)

    else:
        raise TypeError("inputs must match those for either n_periods_between_timestamps or n_periods_between_times")


def ctimestep_to_rtimestep(unit_t, return_status=False):
    """
    Convert character time step to minutes and number time step

    Parameters
    ----------
    unit_t : str
        character time step

    return_status : bool default False
        flag to determine if to return the status

    Returns
    -------
    tuple[int, float]
        if return_status=False timestep in minutes and timestep

    tuple[int, float, int]
        if return_status=True timestep in minutes, timestep and status
    """
    this_procedure = MODNAME + "CTimeStep_To_RTimeStep"

    if return_status:
        status = 0

    deltat_inminutes = 0
    deltat = 1.0

    timestep_index = is_time_interval_valid(unit_t)

    # check for None. python treats 0 as false, so a valid index of zero would be treated as an error
    if timestep_index is None:
        if return_status:
            set_last_message(
                f"{unit_t} is not a recognized time step", FATAL, this_procedure
            )
            status = -1
        else:
            log_message(
                f"{unit_t} is not a recognized time step", FATAL, this_procedure
            )
    else:
        deltat_inminutes = RECOGNIZED_INTERVALS_INMINUTES[timestep_index]

    if return_status:
        return deltat_inminutes, deltat, status
    else:
        return deltat_inminutes, deltat


def time_interval_conversion(to_interval, from_interval):
    """
    Compute conversion factor between two intervals

    Parameters
    ----------
    to_interval : str
        time interval to convert to

    from_interval : str
        time interval to convert from

    Returns
    -------
    float
        conversion factor between time intervals
    """
    this_procedure = MODNAME + "TimeIntervalConversion"

    # convert from_interval from string to minutes
    from_interval_inminutes, _, error_code = ctimestep_to_rtimestep(
        upper_case(from_interval), return_status=True
    )

    if error_code != 0:
        log_message(
            f"{upper_case(from_interval)} is not a valid time interval",
            FATAL,
            this_procedure,
        )

    # convert from_interval from string to minutes
    to_interval_inminutes, _, error_code = ctimestep_to_rtimestep(
        upper_case(to_interval), return_status=True
    )

    if error_code != 0:
        log_message(
            f"{upper_case(to_interval)} is not a valid time interval",
            FATAL,
            this_procedure,
        )

    conversion_factor = to_interval_inminutes / from_interval_inminutes

    return conversion_factor
