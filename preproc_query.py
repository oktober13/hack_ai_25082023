import re


def replacer(test):
    replace_dict = {'\(': '', '\)': '', 'ул\.': 'улица', 'д\.': 'дом ', 'г\.': 'город', 'пер\.': 'переулок',
                    'пр\.': 'проспект',
                    'пр-кт\.': 'проспект', 'ш\.': 'шоссе', 'наб\.': 'набережная', 'пос\.': 'поселок', 'кан\.': 'канал',
                    'пл\.': 'площадь', " г,": " город,", " ул,": " улица,", " пр-кт,": ' проспект,',
                    " б-р,": ' бульвар,',
                    " г ": " город ", " ул ": " улица ", " наб,": " набережная,", " пр,": " проспект,",
                    " пр ": " проспект ",
                    ' д ': ' дом ', " д,": " дом,"}
    test["address"] = test["address"].replace(replace_dict, regex=True)
    test['address'] = test['address'].apply(lambda x: re.sub(r'(?<=\d)к(?=\d)', ', корпус ', x))
    test['address'] = test['address'].str.strip().str.replace(r'\s*,\s*', ', ', regex=True)
    test['address'] = test['address'].apply(lambda x: re.sub(r"(?i)(Строение )([А-Яа-яA-Za-z])", r"литера \2", x))
    test['address'] = test['address'].str.replace('Строение', 'строение')

    def streets_address(address):
        if "улица" in address:
            parts = address.split(', ')
            street_part = [part for part in parts if "улица" in part][0]
            parts.remove(street_part)
            street_name = street_part.replace("улица", "").strip()
            new_address = ', '.join(parts[:2] + ['улица ' + street_name] + parts[2:])
            return new_address
        else:
            return address

    test['address'] = test['address'].apply(streets_address)
    test['address'] = test['address'].apply(lambda x: x + ', литера А' if x[-1].isdigit() else x)
    return test
