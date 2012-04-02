using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Threading;
using Microsoft.Xna;
using Microsoft.Xna.Framework;
using System.IO;
using System.Xml.Serialization;


namespace Misc {
	
	static public class Core {

		/// <summary>
		/// Returns service byt type.
		/// </summary>
		/// <typeparam name="ServiceType"></typeparam>
		/// <param name="game"></param>
		/// <returns></returns>
		public static ServiceType GetService<ServiceType>(this Game game) 
		{
			return (ServiceType)game.Services.GetService(typeof(ServiceType));
		}


		/// <summary>
		/// Adds service and component. Sets update order according to add order.
		/// </summary>
		/// <param name="game"></param>
		/// <param name="component"></param>
		public static void AddServiceAndComponent ( this Game game, GameComponent component )
		{
			game.Components.Add( component );
			int order = game.Components.Count;
			component.UpdateOrder = order;
			game.Services.AddService( component.GetType(), component );
		}



		static public void InitEnvironment()
		{
			Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
		}



		static public void SaveToXml<T> ( T t, string fileName ) 
		{
			XmlSerializer serializer = new XmlSerializer(typeof(T));
			TextWriter textWriter = new StreamWriter(fileName);
			serializer.Serialize(textWriter, t);
			textWriter.Close();
			textWriter.Dispose();
		}



		static public T LoadFromXml<T>(string fileName, bool throwException = true) where T : new() 
		{
			try {
				XmlSerializer serializer = new XmlSerializer(typeof(T));
				TextReader textReader = new StreamReader(fileName);
				T t = (T)serializer.Deserialize(textReader);
				textReader.Close();
				return t;
			} catch ( Exception ) {
				if (throwException) {
					throw;
				}
				T t = new T();
				return t;
			}
		}
	}
}
