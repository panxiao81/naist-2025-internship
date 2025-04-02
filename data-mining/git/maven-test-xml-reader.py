import xml.etree.ElementTree as ET

tree = ET.parse("/mnt/c/Users/pan-x/IdeaProjects/commons-math/commons-math-legacy/target/surefire-reports/TEST-org.apache.commons.math4.legacy.analysis.differentiation.DerivativeStructureTest.xml")
print(tree.getroot().attrib)