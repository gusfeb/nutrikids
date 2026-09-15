CREATE DATABASE IF NOT EXISTS nutrikids_db;
USE nutrikids_db;

CREATE TABLE siswa (
    nis            VARCHAR(20)  PRIMARY KEY,
    nama_lengkap   VARCHAR(100) NOT NULL
);

CREATE TABLE riwayat_konsumsi (
    id_riwayat          INT AUTO_INCREMENT PRIMARY KEY,
    nis_siswa           VARCHAR(20)   NOT NULL,
    tanggal_waktu        DATETIME     NOT NULL,
    path_foto            VARCHAR(255) NOT NULL,
    status_karbohidrat   FLOAT        NOT NULL,  -- skor probabilitas makanan_pokok
    status_protein       FLOAT        NOT NULL,  -- skor probabilitas lauk_pauk
    status_sayur         FLOAT        NOT NULL,
    status_buah          FLOAT        NOT NULL,
    status_susu          FLOAT        NOT NULL,
    FOREIGN KEY (nis_siswa) REFERENCES siswa(nis)
);

-- contoh data siswa untuk testing
INSERT INTO siswa (nis, nama_lengkap) VALUES
('1234', 'Yura'),
('1235', 'Made Arimbawa');
