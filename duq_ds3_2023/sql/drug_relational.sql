CREATE TABLE IF NOT EXISTS company (
  id INTEGER PRIMARY KEY,
  name VARCHAR(250)
);
    
CREATE TABLE IF NOT EXISTS drug (
  id INTEGER PRIMARY KEY,
  name VARCHAR(250) UNIQUE NOT NULL,
  proprietary_name VARCHAR(250) NOT NULL,
  company_id INTEGER REFERENCES drug(id) NOT NULL
);
  
CREATE TABLE IF NOT EXISTS raw_drug (
  applicant VARCHAR(250) NOT NULL,
  proper_name VARCHAR(250) NOT NULL,
  proprietary_name VARCHAR(250)
);
  
INSERT INTO company (name)
SELECT DISTINCT applicant
FROM raw_drug;

INSERT INTO company (name)
SELECT applicant
FROM raw_drug
ON CONFLICT DO NOTHING;

INSERT INTO drug (name, proprietary_name, company_id)
SELECT proper_name, proprietary_name, co.id
FROM raw_drug raw
JOIN company co	
    ON raw.applicant = co.name;