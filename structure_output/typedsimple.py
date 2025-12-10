from typing import TypedDict
class Person(TypedDict):
    name: str
    age: int
    occupation: str

new_person: Person = {
    "name": "Alice",
    "age": '30',
    "occupation": "Engineer"
}
print(new_person)  # it will also print even if age is string   