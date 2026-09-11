"""Synthetic mini-textbook fixture.

Small enough to read, structured exactly like the real book, so the full
ingestion pipeline — sections, exercises, official solutions, glossary — can be
exercised in tests without downloading or parsing the 285-page PDF.
"""

from __future__ import annotations

MINI_BOOK_PAGES: list[dict[str, object]] = [
    {
        "page": 1,
        "text": "Health Information Systems\nA Miniature Test Edition\n",
    },
    {
        "page": 2,
        "text": (
            "Chapter 1\n"
            "Introduction to Health Information Systems\n"
            "1.1 Motivation\n"
            "Health information systems support the processing of data, information and "
            "knowledge in health care settings. A health information system comprises all "
            "computer-based and paper-based components that record, store and provide "
            "information. It is therefore more than the software a hospital happens to buy. "
            "The purpose of managing such a system is to ensure that the right information "
            "reaches the right person at the right time.\n"
        ),
    },
    {
        "page": 3,
        "text": (
            "1.2 Data, Information, and Knowledge\n"
            "Data are raw signs such as a measured value. Information is data interpreted in "
            "a context, for example a measurement attributed to a named patient. Knowledge "
            "is general and holds beyond one individual case, such as the rule that a "
            "sustained elevated blood pressure indicates hypertension requiring treatment. "
            "Distinguishing the three explains why storing more data does not by itself "
            "produce more usable information for clinical work.\n"
        ),
    },
    {
        "page": 4,
        "text": (
            "1.3 Interoperability and Integration\n"
            "Interoperability is the ability of two application components to exchange data "
            "and to use the data that has been exchanged. Technical interoperability covers "
            "the transport of messages, while semantic interoperability requires that the "
            "receiving component interprets coded values as the sender intended. A "
            "communication server reduces the number of point-to-point interfaces by "
            "routing and transforming messages between application components. Integration "
            "is the resulting property of the information system, not the tool that "
            "achieves it.\n"
        ),
    },
    {
        "page": 5,
        "text": (
            "1.4 Exercises\n"
            "1.4.1 Data, Information, and Knowledge\n"
            "A physician reads the note: Diagnosis hypertension, last measurement 160/100 "
            "mmHg. Use this example to explain the difference between data, information "
            "and knowledge.\n"
            "1.4.2 Communication Server\n"
            "Explain which problem a communication server solves in a hospital that runs "
            "twelve application components, and name one disadvantage of introducing one.\n"
        ),
    },
    {
        "page": 6,
        "text": (
            "Solutions to Exercises\n"
            "Chapter 1: Introduction\n"
            "Exercise 1.4.1 Data, Information, and Knowledge\n"
            "The strings 160 and 100 are data because they cannot be interpreted without "
            "context. The information is that this named patient has a recorded blood "
            "pressure of 160/100 mmHg today. The knowledge is the general clinical rule "
            "that such a value indicates hypertension that ought to be treated.\n"
            "Exercise 1.4.2 Communication Server\n"
            "A communication server replaces point-to-point interfaces with a star pattern, "
            "so the number of interfaces grows linearly rather than quadratically with the "
            "number of application components. The disadvantage is that the communication "
            "server becomes a single point of failure that must itself be operated and "
            "monitored around the clock.\n"
        ),
    },
    {
        "page": 7,
        "text": (
            "Glossary\n"
            "Interoperability Ability of two application components to exchange data and to "
            "use the data that has been exchanged.\n"
            "Hypertension Persistently elevated arterial blood pressure that is considered "
            "to require treatment.\n"
        ),
    },
    {
        "page": 8,
        "text": "Index\nInteroperability, 4\nKnowledge, 3\n",
    },
]
