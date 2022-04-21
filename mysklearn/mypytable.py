##############################################
# Programmer: Milo Libby
# Class: CptS 322-01, Spring 2021
# Programming Assignment #2
# 2/10/2022
# Description: This program is represents the MyPyTable class
##############################################

import copy
import csv
from numpy import subtract

from tabulate import tabulate

# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #  """Prints the table in a nicely formatted grid structure. """
    #   print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0])

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        if type(col_identifier) is str:
            index = self.column_names.index(col_identifier)
        else:
            index = col_identifier

        for row in self.data:
            if row[index] == "NA" and include_missing_values == False:
                pass
            else:
                col.append(row[index])
        return col

    def pop_column(self, col_name):
        index = self.column_names.index(col_name)
        col = []
        self.column_names.pop(index)

        for row_num, row in enumerate(self.data):
            value = row[index]
            del self.data[row_num][index]
            col.append(value)
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j in range(len(row)):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass

    def get_subtable(self, cols_to_get):
        """
        Returns a MyPyTable with the cols in cols_to_get
        """
        subtable = MyPyTable()
        subtable.data = []
        subtable.column_names = []
        col_indices = []
        for col_name in cols_to_get:
            subtable.column_names.append(col_name)
            col_indices.append(self.column_names.index(col_name))

        for row in self.data:
            new_row = []
            for index in col_indices:
                new_row.append(row[index])
            subtable.data.append(new_row)

        return subtable

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True)
        for index in row_indexes_to_drop:
            try:
                self.data.pop(index)
            except ValueError:
                pass

    def discretize_col(self, cols, discretizer):
        col_indices = []
        for col_name in cols:
            col_indices.append(self.column_names.index(col_name))

        for i, row in enumerate(self.data):
            for j in range(len(row)):
                if j in col_indices:
                    self.data[i][j] = discretizer(self.data[i][j])

    def drop_rows_with_zero_in_col(self, cols_to_check):
        col_indices = []
        drop_rows = []
        for col_name in cols_to_check:
            col_indices.append(self.column_names.index(col_name))

        for i, row in enumerate(self.data):
            for index in col_indices:
                if row[index] == 0.0 or row[index] == 0:
                    drop_rows.append(i)

        self.drop_rows(drop_rows)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")
        csvreader = csv.reader(infile)
        self.column_names = next(csvreader)

        for row in csvreader:
            self.data.append(row)
        self.convert_to_numeric()

        infile.close()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)

        for row in self.data:
            writer.writerow(row)

        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        duplicate_rows = []   # list of indexes that will be returned
        col_indexes = []
        for col_name in key_column_names:
            col_indexes.append(self.column_names.index(col_name))
            duplicates = []                                 # list of values already read

        for row_index, row in enumerate(self.data):
            values = []
            for col_index in col_indexes:
                values.append(row[col_index])
            if values in duplicates and row_index not in duplicate_rows:  # if it is a duplicate
                duplicate_rows.append(row_index)    # add the index to the list
            else:
                # if its a new value add it to the list of values
                duplicates.append(values)

        return duplicate_rows

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows_to_drop = []
        for row in self.data:
            for value in row:
                if value == "NA":
                    index = self.data.index(row)
                    if index not in rows_to_drop:
                        rows_to_drop.append(index)
        self.drop_rows(rows_to_drop)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        # we do not want indexes with missing values
        col_values = self.get_column(col_index, False)
        col_average = (sum(col_values) / len(col_values))  # calculates average

        for row_index, row in enumerate(self.data):
            value = row[col_index]
            if value == "NA":
                self.data[row_index][col_index] = col_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []

        for instance_number, col_name in enumerate(col_names):
            col_index = self.column_names.index(col_name)
            values = []
            for row in self.data:
                values.append(row[col_index])
            if values == []:
                break
            # attribute column
            stats_data.append(list(col_name))
            # min column
            stats_data[instance_number].append(min(values))
            # max column
            stats_data[instance_number].append(max(values))
            # mid column
            mid = (max(values) + min(values)) / 2
            stats_data[instance_number].append(mid)
            # avg column
            stats_data[instance_number].append(sum(values) / len(values))
            # median column
            values.sort()
            if len(values) % 2 == 0:  # even number of values
                mid1 = values[len(values) // 2]
                mid2 = values[(len(values) // 2) - 1]
                median = (mid1 + mid2) / 2
            else:
                median = values[len(values) // 2]
            stats_data[instance_number].append(median)

        return MyPyTable(stats_header, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # join the headers
        new_header = copy.deepcopy(self.column_names)
        for header_name in other_table.column_names:
            if header_name not in self.column_names:
                new_header.append(header_name)

        joined_data = []
        for row_left in self.data:
            for row_right in other_table.data:
                values_left = []
                values_right = []
                for key_name in key_column_names:
                    # get indexes of keys
                    key_index_left = self.column_names.index(key_name)
                    key_index_right = other_table.column_names.index(key_name)

                    values_right.append(row_right[key_index_right])
                    values_left.append(row_left[key_index_left])

                if values_left == values_right:
                    # prevents both keys from being added to new table
                    values_right_to_add = []
                    for col_num, value in enumerate(row_right):
                        if other_table.column_names[col_num] not in key_column_names:
                            values_right_to_add.append(value)
                    # adding joined row
                    joined_data.append(row_left + values_right_to_add)

        return MyPyTable(new_header, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_data = []
        # join the headers
        # header format will be left columns + (right columns - key columns)
        new_header = []
        for header_name in self.column_names:
            new_header.append(header_name)

        for header_name in other_table.column_names:
            if header_name not in self.column_names:
                new_header.append(header_name)

        other_left_indexes = []
        other_right_indexes = []
        # list indexes of columns without the keys for both tables
        for col_name in self.column_names:
            if col_name not in key_column_names:
                other_left_indexes.append(self.column_names.index(col_name))

        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                other_right_indexes.append(
                    other_table.column_names.index(col_name))

        # get indexes of keys
        left_indexes = []
        right_indexes = []

        for col_name in key_column_names:
            left_indexes.append(self.column_names.index(col_name))
            right_indexes.append(other_table.column_names.index(col_name))

        # outer join left
        for row_left in self.data:
            match = False
            for row_right in other_table.data:
                values_left = []  # list of values of key columns for the left table
                values_right = []  # list of values of key columns for the right table

                for index in left_indexes:
                    values_left.append(row_left[index])
                for index in right_indexes:
                    values_right.append(row_right[index])

                if values_right == values_left:  # if the key values match up
                    values_right_to_add = []  # non key values in the row from the right table

                    for index in other_right_indexes:
                        values_right_to_add.append(row_right[index])

                    joined_data.append(row_left + values_right_to_add)
                    match = True

            if match == False:
                values_right_to_add = []
                for i in range(len(other_right_indexes)):
                    values_right_to_add.append("NA")

                joined_data.append(row_left + values_right_to_add)

        row_left_copy = []

        # outer join right
        for row_right in other_table.data:
            for row_left in self.data:
                match = False
                values_left = []
                values_right = []
                row_left_copy = copy.deepcopy(row_left)

                for index in left_indexes:
                    values_left.append(row_left[index])
                for index in right_indexes:
                    values_right.append(row_right[index])

                for index in other_right_indexes:
                    values_right_to_add = []
                    values_right_to_add.append(row_right[index])

                if values_right == values_left:
                    values_left_to_add = []

                    for index in other_left_indexes:
                        values_left_to_add.append(row_left[index])
                    match = True
                    break

            if match == False:
                values_left_to_add = []
                for i in range(len(other_left_indexes)):
                    values_left_to_add.append("NA")

                for i, col_nameR in enumerate(other_table.column_names):
                    for j, col_nameL in enumerate(self.column_names):
                        if col_nameR == col_nameL:
                            row_left_copy[j] = row_right[i]
                        if col_nameL not in key_column_names:
                            row_left_copy[j] = "NA"

                joined_data.append(
                    (row_left_copy + values_right_to_add))

        return MyPyTable(new_header, joined_data)

    def __str__(self) -> str:
        string = ""
        for col in self.column_names:
            print(col, end=" ")
        print()

        for row in self.data:
            for value in row:
                string += str(value) + " "
            string += "\n"
        return string

    def print_data(self):
        print(tabulate(self.data, self.column_names))

    """
      takes in a column name 
      returns a dictionary of each attribute with its frequency
    """

    def get_frequencies_categorical(self, column_name):
        values = self.get_column(column_name, False)

        values_with_frequencies_dictionary = {}
        for value in values:
            if value not in values_with_frequencies_dictionary:
                values_with_frequencies_dictionary[value] = 1
            else:
                values_with_frequencies_dictionary[value] += 1

        return values_with_frequencies_dictionary


# TODO: copy your mypytable.py solution from PA2-PA3 here
