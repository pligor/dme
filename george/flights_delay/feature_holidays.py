import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureHolidays(object):
    def __init__(self, data_frame):
        self.df = data_frame.copy()

        self.le_dict = dict()  # Initialise an empty dictionary to keep all LabelEncoders
        self.df_hol = data_frame.copy(deep=True)  # Make a copy of the DataFrame
        le = LabelEncoder().fit(data_frame['FL_DATE'])  # Initialise the LabelEncoder and fit
        self.df_hol['FL_DATE'] = le.transform(data_frame['FL_DATE'])  # Transform data and save in credit_clean DataFrame
        self.le_dict['FL_DATE'] = le  # Store the LabelEncdoer in dictionary
        # increase the day by one because Label Encoder starts from 0

    def transform(self):
        df_hol['IS_HOLIDAY'] = self.getBinaryHolidayFeature()

    def getBinaryHolidayFeature(self):
        # make a binary holiday feature (if on holiday or not)
        isholiday = np.zeros(self.df_hol.shape[0])
        ii = 0
        for day in self.df_hol['FL_DATE']:
            if day in self.holidays:
                isholiday[ii] = 1
            ii += 1

        return isholiday

    # source: http://www.officeholidays.com/calendars/year_planner/index.php?planner_year=2016&planner_country=USA
    # Jan 01: New Years Day
    # Jan 18: Martin Luther King Day
    # Feb 12: Lincoln's Birthday (Only New York)
    # Feb 15: Presidents' Day
    # Apr 15: Emancipation Day
    # May 08: Mother's Day
    # May 30: Memorial Day
    # Jun 19: Father's Day
    # Jul 04: Independence Day
    # Sep 05: Labor Day
    # Oct 10: Columbus Day
    # Nov 11: Veterans Day
    # Nov 24: Thanksgiving
    # Nov 25: Day after Thanksgiving
    # Dec 26: Christmas Day (in lieu)
    # pick: Jan1, Jan18, Feb12, Apr15, May30, Jul4, Sep05, Oct10, Nov11, Nov24, Nov25, Dec26
    holidays = [0, 17, 42, 105, 150, 165, 248, 283, 315, 328, 329, 360]

