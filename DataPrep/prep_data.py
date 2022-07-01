import re
import sys
import langid
langid.set_languages(['en','es'])

try:
    all = open("../PreppedData/all_data.csv", "w")
    span = open("../PreppedData/spanish.csv", "w")
    engl = open("../PreppedData/english.csv", "w")
except OSError as e:
    print ("Can't open file for writing")
    exit()

all.write("text,label\n")
span.write("text,label\n")
engl.write("text,label\n")

def main(files):

    student_langs = {}
    langs = re.compile(r"^.*\|(ST\d\d\d).*?\|([a-zA-Z]*)_(:?speaker|student)\|(.*$)")
    convo_start = re.compile(r"^\*(ST\d\d\d):\s*(.*)$")
    continuation = re.compile(r"^\s(\S*.*\S*)\s*$")


    for filename in files:
        try:
            f = open(filename, "r", encoding="utf-8")
        except FileNotFoundError as e:
            print("Can't open", filename)
            return

        convo_line = ""
        for line in f:
            l = langs.match(line.rstrip())
            if l:
                student = l.group(1)
                language = l.group(2).lower()
                if not student in student_langs:
                    student_langs[student] = language
                else:
                    if student_langs[student] != language:
                        print("UH OH: Language mismatch:", student)
            else:
                l = convo_start.match(line.rstrip())
                if l:
                    if len(convo_line) > 0:
                        write_line(student, convo_line, student_langs)
                        convo_line = ""

                    student = l.group(1)
                    convo_line = l.group(2)

                l = continuation.match(line.rstrip())
                if l:
                    convo_line += " " + l.group(1).strip()

clean_text = re.compile(r"(<|>|\[|\]|\&|\=|laughs)")
def write_line(student, quote, student_langs):
    clean_quote = re.sub(clean_text, "", quote)
    sp1 = re.compile(r"^¿.*\?$")
    if sp1.match(clean_quote):
        lang = "es"
    else:
        lang = langid.classify(clean_quote)[0]
    l1 = None
    if lang == "en":
        if student_langs[student] == "english":
            l1 = "1"
        elif student_langs[student] == "spanish":
            l1 = "0"
        else:
            print("Inconsistent language result")
    elif lang == "es":
        if student_langs[student] == "spanish":
            l1 = "1"
        elif student_langs[student] == "english":
            l1 = "0"
        else:
            print("Inconsistent language result")

    print(f"\"{quote}\",{l1}")
    if lang == "en":
        engl.write(f"\"{quote}\",{l1}\n")
    if lang == "es":
        span.write(f"\"{quote}\",{l1}\n")
    all.write(f"\"{quote}\",{l1}\n")



main(sys.argv[1:])

