"""
Parse and populate the fireballs detection results into a database. The database is used by
a web application for displaying the results.

"""
from __future__ import print_function

import time
import sqlite3
import pandas as pd
import settings as s

# Database
DB_NAME = 'report/app.db'
# Database tables
FIREBALL_TABLE = 'transient'


def main(): 
    start_time = time.time()

    print('\nPopulate results into database:')

    # Open database connection  
    conn = sqlite3.connect(DB_NAME)
 
    # Read the results file
    df = pd.read_csv(s.RESULTS_FILE)
    print('  file: %s' % (s.RESULTS_FILE))
    print('  # records: %d' % (len(df)))

    # Insert the results into a sqlite database
    df.to_sql(FIREBALL_TABLE, conn, if_exists='replace', index=False)

    # Close the database connection
    conn.close()
    
    print('\nResults populated successfully: %.3f seconds' % (time.time() - start_time))
    print('Generated file:')
    print('  %s\n' % (DB_NAME))

if __name__ == '__main__':
    main()
