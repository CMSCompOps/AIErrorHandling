<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <title></title>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css"></link>
      </head>
      <body>
        <xsl:for-each select="files/wf">
            <a>
                  <xsl:attribute name="href">
                          <xsl:value-of select="lnk"/>
                  </xsl:attribute>
                      <xsl:value-of select="name"/>
            </a>
            <br />
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
