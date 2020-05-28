<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <title></title>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css"></link>
      </head>
      <body>
        <table class="w3-table-all w3-hoverable" id="table_wfs">
          <tr>
            <th>Workflow</th>
            <th>action</th>
            <th>description</th>
            <th>error</th>
            <th>unified</th>
            <th>.json</th>
            <th>console</th>
          </tr>
          <xsl:for-each select="files/wf">
            <xsl:sort select="action" order="descending"/>
            <xsl:sort select="main_err" order="descending"/>
            <tr>
              <td>
                <a>
                  <xsl:attribute name="href">
                    <xsl:value-of select="lnk"/>
                  </xsl:attribute>
                  <xsl:value-of select="name"/>
                </a>
              </td>
              <td><xsl:value-of select="action"/></td>
              <td><xsl:value-of select="action_description"/></td>
              <td><xsl:value-of select="main_err"/></td>
              <td>
                <a>
                  <xsl:attribute name="href">
                    <xsl:value-of select="unified"/>
                  </xsl:attribute>
                  unified
                </a>
              </td>
              <td>
                <a>
                  <xsl:attribute name="href">
                    <xsl:value-of select="json"/>
                  </xsl:attribute>
                  json
                </a>
              </td>
              <td>
                <a>
                  <xsl:attribute name="href">
                    <xsl:value-of select="console"/>
                  </xsl:attribute>
                  console
                </a>
              </td>
            </tr>
          </xsl:for-each>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
