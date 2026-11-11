"""Surya2 layout labels emitted by the model + canonicalization to surya's
public label vocabulary."""

# Canonical text-bearing labels — used by blank-region filters to decide
# which blocks may be dropped when their underlying image region is empty.
# Excludes Picture/Figure/Diagram/Table/Form/Equation/etc., which can legitimately
# contain whitespace or solid fills.
TEXT_LABELS = frozenset(
    {
        "Text",
        "SectionHeader",
        "PageHeader",
        "PageFooter",
        "Caption",
        "Footnote",
        "Code",
        "Bibliography",
    }
)


# Canonicalize raw model labels to public surya label names. Marker and other
# downstream consumers depend on these names.
LAYOUT_PRED_RELABEL = {
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Equation-Block": "Equation",
    "List-Group": "ListGroup",
    "Page-Header": "PageHeader",
    "Page-Footer": "PageFooter",
    "Image": "Picture",
    "Section-Header": "SectionHeader",
    "Table": "Table",
    "Text": "Text",
    "Complex-Block": "Figure",
    "Code-Block": "Code",
    "Form": "Form",
    "Table-Of-Contents": "TableOfContents",
    "Figure": "Figure",
    "Chemical-Block": "ChemicalBlock",
    "Diagram": "Diagram",
    "Bibliography": "Bibliography",
    "Blank-Page": "BlankPage",
}
