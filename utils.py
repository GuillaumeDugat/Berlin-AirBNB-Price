from datetime import datetime

def convert_date(date_str: str) -> int:
 
    date_compare = datetime(2022, 10, 13)
    date_to_compare = datetime.strptime(date_str, '%Y-%m-%d')
    comparison = (date_compare - date_to_compare).days

    return comparison
