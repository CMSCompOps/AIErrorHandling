<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <title>
          <xsl:value-of select="workflow/name" />
        </title>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css"></link>
      </head>
      <body>
        <dl>
          <dt>workflow name:</dt>
          <dd>- <xsl:value-of select="workflow/name"/> </dd>
          <dt>recommended action:</dt>
          <dd>- <xsl:value-of select="workflow/summary/action"/> </dd>
          <dt>xrootd:</dt>
          <dd>- <xsl:value-of select="workflow/summary/xrootd"/> </dd>
          <dt>main error</dt>
          <dd>- <xsl:value-of select="workflow/summary/main_error"/> </dd>
          <dt>notes</dt>
          <dd>- <xsl:value-of select="workflow/summary/description"/> </dd>
        </dl>

        <table class="w3-table-all w3-hoverable" id="table_wfs">
          <tr class="w3-red">
            <th> error </th>
            <th> count </th>
            <th> ratio </th>
            <th> good sites </th>
            <th> bad sites </th>
          </tr>
          <xsl:for-each select="workflow/errors/error">
            <xsl:sort select="count" data-type="number" order="descending" />
            <tr>
              <td><xsl:value-of select="id"/></td>
              <td><xsl:value-of select="count"/></td>
              <td><xsl:value-of select="ratio"/></td>
              <td>
                <ul>
                  <xsl:for-each select="good_sites/site">
                    <li>
                      <xsl:value-of select="name" />
                    </li>
                  </xsl:for-each>
                </ul>
              </td>
              <td>
                <ul>
                  <xsl:for-each select="bad_sites/site">
                    <li>
                      <xsl:value-of select="name" />
                    </li>
                  </xsl:for-each>
                </ul>
              </td>
            </tr>
          </xsl:for-each>
        </table>

        <table class="w3-table-all w3-hoverable" id="table_tasks">
          <tr class="w3-red">
            <th> task </th>
            <th> xrootd </th>
            <th> good sites </th>
            <th> bad sites </th>
          </tr>
          <xsl:for-each select="workflow/tasks/task">
            <tr>
              <td><xsl:value-of select="name"/></td>
              <td><xsl:value-of select="xrootd"/></td>
              <td>
                <ul>
                  <xsl:for-each select="good_sites/site">
                    <li>
                      <xsl:value-of select="name" />
                    </li>
                  </xsl:for-each>
                </ul>
              </td>
              <td>
                <ul>
                  <xsl:for-each select="bad_sites/site">
                    <li>
                      <xsl:value-of select="name" />
                    </li>
                  </xsl:for-each>
                </ul>
              </td>
            </tr>
          </xsl:for-each>
        </table>


      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
