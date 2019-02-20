/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
 EXPORT Name := 'KMeans';
 EXPORT Description := 'KMeans Bundle for Clustering algorithm';
 EXPORT Authors := ['HPCCSystems'];
 EXPORT License := 'http://www.apache.org/licenses/LICENSE-2.0';
 EXPORT Copyright := 'Copyright (C) 2019 HPCC Systems';
 EXPORT DependsOn := ['ML_Core 3.2.2'];
 EXPORT Version := '1.0.0';
 EXPORT PlatformVersion := '6.4.0';
END;